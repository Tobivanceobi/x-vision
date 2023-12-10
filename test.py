import hashlib

def sha256_hash(data):
    return hashlib.sha256(data).hexdigest()

def length_extension_attack(original_hash, original_message, additional_data, key_length):
    # This is a placeholder function to demonstrate the concept
    # In a real scenario, here you would manipulate the original hash and message
    # to calculate the new hash with the additional data
    new_message = original_message + additional_data
    fake_key = 'A' * key_length
    new_hash = sha256_hash((fake_key + new_message).encode())

    return new_hash, new_message

# Example usage
original_hash = 'aa0e162320f7c82eb469edefba0b1ac8e06a89e72c7b4e45837cee6df0994273'  # Replace with the hash you obtained
key_length = 12
original_message = 'tobias.ettling@student.unisg.ch'
additional_data = '&role=admin'

new_hash, new_message = length_extension_attack(original_hash, original_message, additional_data, key_length)

print(f"New Hash: {new_hash}")
print(f"New Message: {new_message}")
