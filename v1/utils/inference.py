from fastapi import FastAPI
from pydantic import BaseModel

class RequestType(BaseModel):
    path:str
    

class Inference:

  

  def __init__(self):
    pass

  def load_test_data(self,path:str):
    df = pd.read_csv(path)
    df.rename(columns={"#1 String":"text_1","#2 String":"text_2"},inplace = True)
    df = df[["text_1","text_2"]]
    return df

  def get_preds(self,input_ids_test,attention_mask_test):
    self.load_model("saved_weights.pt")
    with torch.no_grad():
      preds = model(input_ids_test.to(device), attention_mask_test.to(device))
      preds = preds.detach().cpu().numpy()
    return preds

  
  def load_model(self,model_path="saved_weights.pt"):
    model.load_state_dict(torch.load(path))

app = FastAPI() 
@app.post("/prediction")
def make_preds(path:RequestType):
  inf = Inference()
  df = inf.load_test_data(path)
  print(df.columns)
  
  preprocess = bert_preprocess()
  input_ids_test,attention_mask_test = preprocess.create_tokenizer_list(df,
                                                                        "text_1",
                                                                        "text_2",
                                                                        max_length)
  #convert to tensor
  input_ids_test = torch.tensor(input_ids_test,dtype=torch.long)
  attention_mask_test = torch.tensor(attention_mask_test,dtype=torch.long)
  # get predictions for test data
  preds = inf.get_preds(input_ids_test,attention_mask_test)
  preds = np.argmax(preds, axis = 1)
  preds = preds.tolist()
  df["Quality"] = preds
  return df
