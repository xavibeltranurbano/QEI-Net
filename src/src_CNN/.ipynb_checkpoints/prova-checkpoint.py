import tensorflow as tf

# List all available devices visible to TensorFlow
print("Available devices:")
devices = tf.config.list_physical_devices()
for device in devices:
    print(device)

# Specifically check for GPU availability in TensorFlow
print("\nIs GPU available: ", tf.test.is_gpu_available())

# Alternatively, list only GPU devices (if any)
print("\nList of GPU devices: ")
print(tf.config.list_physical_devices('GPU'))
