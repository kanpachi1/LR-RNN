import argparse
import json
import os

from ignite.engine import (
    Engine,
    Events,
)
from ignite.handlers import (
    EarlyStopping,
    ModelCheckpoint,
    global_step_from_engine,
)
from ignite.handlers.tensorboard_logger import (
    OutputHandler,
    TensorboardLogger,
)
from ignite.handlers.tqdm_logger import ProgressBar
import torch
import yaml

from lrrnn import info
from lrrnn import data_loader
from lrrnn.feature_extraction import (
    extract_char_from_surface,
    extract_chartype_from_surface,
    extract_features,
)
from lrrnn.metrics import F1
from lrrnn.models import (
    WordSegmentationRNN,
    create_ws_trainer,
)
from lrrnn.tokenizers import RNNTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to yaml file containing configuration")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["savedir"])

    torch.manual_seed(info.seed)
    device = torch.device("cuda")

    print("Building data loader...")
    unsegmented_sentences, labels = data_loader.load_corpus_data_for_word_segmentation(
        config["train"]
    )

    char1_features, char1_ttoi = extract_features(
        unsegmented_sentences, extract_char_from_surface, n_gram=1, window_size=3
    )
    char2_features, char2_ttoi = extract_features(
        unsegmented_sentences, extract_char_from_surface, n_gram=2, window_size=3
    )
    char3_features, char3_ttoi = extract_features(
        unsegmented_sentences, extract_char_from_surface, n_gram=3, window_size=3
    )
    type1_features, type1_ttoi = extract_features(
        unsegmented_sentences, extract_chartype_from_surface, n_gram=1, window_size=3
    )
    type2_features, type2_ttoi = extract_features(
        unsegmented_sentences, extract_chartype_from_surface, n_gram=2, window_size=3
    )
    type3_features, type3_ttoi = extract_features(
        unsegmented_sentences, extract_chartype_from_surface, n_gram=3, window_size=3
    )

    train_dl = data_loader.build_data_loader(
        (
            char1_features,
            char2_features,
            char3_features,
            type1_features,
            type2_features,
            type3_features,
        ),
        labels,
        config["batch_size"],
        config["unroll_size"],
    )

    unsegmented_sentences, labels = data_loader.load_corpus_data_for_word_segmentation(
        config["valid"]
    )
    valid_dl = torch.utils.data.DataLoader(
        data_loader.TokenizerEvalDataset(unsegmented_sentences, labels),
        batch_size=1,
        shuffle=False,
    )

    print("Building model...")
    model = WordSegmentationRNN(
        len(char1_ttoi),
        len(char2_ttoi),
        len(char3_ttoi),
        config["char_embedding_dim"],
        len(type1_ttoi),
        len(type2_ttoi),
        len(type3_ttoi),
        config["chartype_embedding_dim"],
        config["hidden_size"],
        2,
        config["dropout"],
    )
    model.to(device)
    tokenizer = RNNTokenizer(
        char1_ttoi,
        char2_ttoi,
        char3_ttoi,
        type1_ttoi,
        type2_ttoi,
        type3_ttoi,
        model,
        device,
    )

    def create_evaluator():

        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                surface, gt_tokens = batch
                # Assume batch_size=1
                surface = surface[0]
                gt_tokens = [t[0] for t in gt_tokens]
                tokens = tokenizer.predict(surface)
                predictions = [[t] for t in tokens]
                groundtruths = [[t] for t in gt_tokens]
                return predictions, groundtruths

        engine = Engine(_inference)
        f1 = F1()
        f1.attach(engine, "f1")
        return engine

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    patience = 4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=patience
    )
    trainer = create_ws_trainer(model, loss_fn, optimizer, device)
    evaluator = create_evaluator()

    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {"loss": x})
    tb_logger = TensorboardLogger(log_dir=config["savedir"])
    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(tag="train", output_transform=lambda x: {"loss": x}),
        event_name=Events.ITERATION_COMPLETED,
    )
    tb_logger.attach(
        evaluator,
        log_handler=OutputHandler(
            tag="valid",
            metric_names=["f1"],
            global_step_transform=global_step_from_engine(trainer),
        ),
        event_name=Events.EPOCH_COMPLETED,
    )

    def score_function(engine):
        return engine.state.metrics["f1"]

    earlystopping_handler = EarlyStopping(
        patience=patience + 4, score_function=score_function, trainer=trainer
    )
    evaluator.add_event_handler(Events.COMPLETED, earlystopping_handler)

    modelcheckpoint_handler = ModelCheckpoint(
        config["savedir"],
        "best",
        score_function=score_function,
        score_name="f1",
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, modelcheckpoint_handler, {"model": model}
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_dl)
        metrics = evaluator.state.metrics
        scheduler.step(metrics["f1"])
        pbar.log_message(
            "Epoch {:3d}: f1={:.6f}, lr={}".format(
                trainer.state.epoch, metrics["f1"], scheduler._last_lr[0]
            )
        )

    trainer.run(train_dl, max_epochs=config["max_epochs"])

    with open(os.path.join(config["savedir"], "other_params.json"), "w") as f:
        json.dump(
            {
                "char1_ttoi": char1_ttoi,
                "char2_ttoi": char2_ttoi,
                "char3_ttoi": char3_ttoi,
                "char_embedding_dim": config["char_embedding_dim"],
                "chartype1_ttoi": type1_ttoi,
                "chartype2_ttoi": type2_ttoi,
                "chartype3_ttoi": type3_ttoi,
                "chartype_embedding_dim": config["chartype_embedding_dim"],
                "hidden_size": config["hidden_size"],
            },
            f,
        )
