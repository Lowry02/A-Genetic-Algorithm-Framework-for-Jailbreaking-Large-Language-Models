class Logger():
  def __init__(self, header: list = None) -> None:
    """_summary_

    Args:
        header (list, optional): CSV file header. Defaults to None.
    """
    
    assert type(header) == list, "header must be a list"
    
    self.header = header
    self.file = None
    
  def create_file(self, file_name:str):
    """_summary_

    Args:
        file_name (str): Name of the file(it will be created).
        
    Description:
        Creates the file and writes the header line.
    """
    
    assert type(file_name) == str, "file_name must be a string"
    
    self.file = open(file_name, 'w')
    header_str = ""
    for col_name in self.header:
      header_str += f"{col_name},"
    header_str = header_str[:-1] + "\n"
    
    self.file.write(header_str)
    
  def close_file(self) -> None:
    self.file.flush()    
    self.file.close()
    self.file = None
    
  def is_initialized(self):
    return self.file != None
    
  def log(self, data:dict):
    """_summary_

    Args:
        data (dict): Dictionary with keys equal to the header names.
    Description:
        Logs the data in the CSV file.
    """
    
    assert type(data) == dict, "data must be a string"
    assert all([col in self.header for col in data.keys()]), "A key in the data keys is not in the header columns."
    
    line_str = ""
    for col_name in self.header:
      if col_name in data.keys():
        value = data[col_name]
      else:
        value = "NaN"
      line_str += f"{value},"
    line_str = line_str[:-1] + "\n"
    
    self.file.write(line_str)
      
    