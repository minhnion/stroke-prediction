import torch
import unittest

from src.models.tabtransformer import TabTransformerModel

class TestTabTransformerModel(unittest.TestCase):

    def test_forward_pass(self):
        CATEGORIES = (10, 5, 3)
        NUM_CONTINUOUS = 4
        BATCH_SIZE = 8
        DIM = 32
        DEPTH = 3
        HEADS = 4
        DIM_OUT = 1

        model = TabTransformerModel(
            categories=CATEGORIES,
            num_continuous=NUM_CONTINUOUS,
            dim=DIM,
            depth=DEPTH,
            heads=HEADS,
            dim_out=DIM_OUT
        )

        x_categ = torch.randint(0, CATEGORIES[0], (BATCH_SIZE, 1))
        x_categ = torch.cat([x_categ, torch.randint(0, CATEGORIES[1], (BATCH_SIZE, 1))], dim=1)
        x_categ = torch.cat([x_categ, torch.randint(0, CATEGORIES[2], (BATCH_SIZE, 1))], dim=1)

        x_cont = torch.randn(BATCH_SIZE, NUM_CONTINUOUS)
        
        print(f"\nInput categorical shape: {x_categ.shape}")
        print(f"Input continuous shape: {x_cont.shape}")


        output = model(x_categ, x_cont)
        
        print(f"Output shape: {output.shape}")
        

        expected_shape = (BATCH_SIZE, DIM_OUT)
        self.assertEqual(output.shape, expected_shape)
        
        self.assertTrue(isinstance(output, torch.Tensor))

if __name__ == '__main__':
    unittest.main()

