import configparser

class Reader():
    def __init__(self, filename):
        self.config = configparser.ConfigParser()

    def __call__(self, key):
        pass
        

# Assuming you have a file named 'config.conf' with the given content
filename = 'config.conf'

# Create a ConfigParser instance
config = configparser.ConfigParser()
config.read(filename)

# Access the parameters
network_name = config.get('Network', 'name')
num_epochs = config.getint('Network', 'num_epochs')
init_rate = config.getfloat('Network', 'init_rate')
loss = config.get('Network', 'Loss')

batch_size = config.getint('Data', 'batchSize')
shape = json.loads(config.get('Data', 'shape'))
data_dir = config.get('Data', 'data_dir')
save_dir = config.get('Data', 'save_dir')

# Print the values
print("Network name:", network_name)
print("Number of epochs:", num_epochs)
print("Initial learning rate:", init_rate)
print("Loss:", loss)

print("Batch size:", batch_size)
print("Shape:", shape)
print("Data directory:", data_dir)
print("Save directory:", save_dir)
