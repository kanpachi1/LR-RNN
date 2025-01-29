import unittest

import torch

from lrrnn.models import WordSegmentationRNN, _ws_train, POSTaggingRNN, _pt_train


class TestWordSegmentation(unittest.TestCase):

    def test_forward(self):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        hidden_size = 4
        num_classes = 2
        model = WordSegmentationRNN(
            10, 10, 10, 32, 10, 10, 10, 8, hidden_size, num_classes
        )
        model.to(device)
        self.assertEqual(model._input_size, 600)

        seq_len = 10
        batch_size = 2
        wsize = 3
        char1 = torch.randint(10, (seq_len, batch_size, wsize * 2), device=device)
        char2 = torch.randint(10, (seq_len, batch_size, wsize * 2 - 1), device=device)
        char3 = torch.randint(10, (seq_len, batch_size, wsize * 2 - 2), device=device)
        chartype1 = torch.randint(10, (seq_len, batch_size, wsize * 2), device=device)
        chartype2 = torch.randint(
            10, (seq_len, batch_size, wsize * 2 - 1), device=device
        )
        chartype3 = torch.randint(
            10, (seq_len, batch_size, wsize * 2 - 2), device=device
        )
        output, (h, c) = model(char1, char2, char3, chartype1, chartype2, chartype3)
        self.assertEqual(output.size(), (seq_len, batch_size, num_classes))
        self.assertEqual(h.size(), (2, batch_size, hidden_size))
        self.assertEqual(c.size(), (2, batch_size, hidden_size))


class TestPOSTagging(unittest.TestCase):

    def test_forward(self):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        hidden_size = 4
        num_classes = 2
        model = POSTaggingRNN(
            10, 10, 10, 32, 10, 10, 10, 8, 50, 100, hidden_size, num_classes
        )
        model.to(device)
        self.assertEqual(model._input_size, 700)

        seq_len = 10
        batch_size = 2
        wsize = 3
        char1 = torch.randint(10, (seq_len, batch_size, wsize * 2), device=device)
        char2 = torch.randint(10, (seq_len, batch_size, wsize * 2 - 1), device=device)
        char3 = torch.randint(10, (seq_len, batch_size, wsize * 2 - 2), device=device)
        chartype1 = torch.randint(10, (seq_len, batch_size, wsize * 2), device=device)
        chartype2 = torch.randint(
            10, (seq_len, batch_size, wsize * 2 - 1), device=device
        )
        chartype3 = torch.randint(
            10, (seq_len, batch_size, wsize * 2 - 2), device=device
        )
        word = torch.randint(50, (seq_len, batch_size), device=device)
        output, (h, c) = model(
            char1, char2, char3, chartype1, chartype2, chartype3, word
        )
        self.assertEqual(output.size(), (seq_len, batch_size, num_classes))
        self.assertEqual(h.size(), (2, batch_size, hidden_size))
        self.assertEqual(c.size(), (2, batch_size, hidden_size))


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available()

    def test__ws_train(self):
        device = torch.device("cuda")
        hidden_size = 4
        num_classes = 2
        model = WordSegmentationRNN(
            10, 10, 10, 32, 10, 10, 10, 8, hidden_size, num_classes
        )
        model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9)
        seq_len = 10
        batch_size = 2
        wsize = 3
        char1 = torch.randint(10, (seq_len, batch_size, wsize * 2), device=device)
        char2 = torch.randint(10, (seq_len, batch_size, wsize * 2 - 1), device=device)
        char3 = torch.randint(10, (seq_len, batch_size, wsize * 2 - 2), device=device)
        chartype1 = torch.randint(10, (seq_len, batch_size, wsize * 2), device=device)
        chartype2 = torch.randint(
            10, (seq_len, batch_size, wsize * 2 - 1), device=device
        )
        chartype3 = torch.randint(
            10, (seq_len, batch_size, wsize * 2 - 2), device=device
        )
        y = torch.randint(num_classes, (seq_len, batch_size))
        batch = (char1, char2, char3, chartype1, chartype2, chartype3, y)

        _ = _ws_train(model, loss_fn, optimizer, device, batch)

    def test__pt_train(self):
        device = torch.device("cuda")
        hidden_size = 4
        num_classes = 2
        model = POSTaggingRNN(
            10, 10, 10, 32, 10, 10, 10, 8, 50, 100, hidden_size, num_classes
        )
        model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9)
        seq_len = 10
        batch_size = 2
        wsize = 3
        char1 = torch.randint(10, (seq_len, batch_size, wsize * 2), device=device)
        char2 = torch.randint(10, (seq_len, batch_size, wsize * 2 - 1), device=device)
        char3 = torch.randint(10, (seq_len, batch_size, wsize * 2 - 2), device=device)
        chartype1 = torch.randint(10, (seq_len, batch_size, wsize * 2), device=device)
        chartype2 = torch.randint(
            10, (seq_len, batch_size, wsize * 2 - 1), device=device
        )
        chartype3 = torch.randint(
            10, (seq_len, batch_size, wsize * 2 - 2), device=device
        )
        word = torch.randint(50, (seq_len, batch_size), device=device)
        y = torch.randint(num_classes, (seq_len, batch_size))
        batch = (char1, char2, char3, chartype1, chartype2, chartype3, word, y)

        _ = _pt_train(model, loss_fn, optimizer, device, batch)
