

class dataset_creator:
  
  
  def __init__(self, task_settings, train_df, test_df, class_num_mapping, semantic_class_vectors):
    
      self.train_df = train_df
      self.test_df = test_df
      self.class_num_mapping = class_num_mapping
      self.task_settings = task_settings
      self.semantic_class_vectors = semantic_class_vectors
      
      self.create_transfer_dataset(task_settings)
      self.compute_semantic_statistics()
        
        
        
  def create_transfer_dataset(self, task_settings):
  
      self.source_train_df = self.train_df.loc[self.train_df['class'].isin(task_settings["source_classes"]["class_num"])].sample(frac=task_settings["source_classes"]["train_fract"][0])
      self.source_test_df = self.test_df.loc[self.test_df['class'].isin(task_settings["source_classes"]["class_num"])].sample(frac=task_settings["source_classes"]["test_fract"][0])
      
      self.target_train_df = self.train_df.loc[self.train_df['class'].isin(task_settings["target_classes"]["class_num"])].sample(frac=task_settings["target_classes"]["test_fract"][0])
      self.target_test_df = self.test_df.loc[self.test_df['class'].isin(task_settings["target_classes"]["class_num"])].sample(frac=task_settings["target_classes"]["test_fract"][0])
      
      print("-- created pandas dataframes with source and target domain split --")

      self.get_features_labels()
      
      
      
  def get_features_labels(self):

      self.source_train_x = np.array(self.source_train_df["content"].tolist())
      self.source_train_y = np.array(self.source_train_df["class"].tolist())
      
      self.source_test_x = np.array(self.source_test_df["content"].tolist())
      self.source_test_y = np.array(self.source_test_df["class"].tolist())
      
      self.target_train_x = np.array(self.target_train_df["content"].tolist())
      self.target_train_y = np.array(self.target_train_df["class"].tolist())
      
      self.target_test_x = np.array(self.target_test_df["content"].tolist())
      self.target_test_y = np.array(self.target_test_df["class"].tolist())
      
      print("-- created numpy arrays with features and labels of source and target domain split --")
      
      
  def compute_semantic_statistics(self):

      #self.semantic_class_vectors
      #self.task_settings

      #train_class_similarity

      #self.train_class_similarity = 1 - spatial.distance.cosine(dataSetI, dataSetII)

      print("-- computed semantic statistics of transfer task (cosine similarity etc.) --")



##______________________________________________________________________
  
  
task_settings = {"source_classes": {"class_num": [1,2], "train_fract": [0.1], "test_fract": [0.1] }, 
                 "target_classes": {"class_num": [3,4], "train_fract": [0.1], "test_fract": [0.1]}}

dataset = dataset_creator(task_settings, train_df, test_df, class_num_mapping, semantic_class_vectors)