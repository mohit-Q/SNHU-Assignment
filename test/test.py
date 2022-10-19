import unittest
import sys
sys.path.append('../')
# print(sys.path)

from v1.utils.bert_preprocess import bert_preprocess

class TestBert(unittest.TestCase):
    
    
    def test_one(self):
        du_list = [101, 2023, 2003, 3131, 102, 4638, 1996, 102]
        bpre = bert_preprocess()
        string1 = "this is unit test"
        string2 = "check the encoding"
        l = 8
        out = bpre.batch_encoder(string1,string2,l)[0]
        
        self.assertNotEqual(out,bpre)
        
        

if __name__ == '__main__':
    unittest.main()


