import json
import hashlib
from datetime import datetime


class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()


# ⛓️ Blockchain Class
class Blockchain:
    def __init__(self, file_path="chain_data.json"):
        self.file_path = file_path
        self.chain = []
        self.load_chain()

        if len(self.chain) == 0:
            self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(
            index=0,
            timestamp=str(datetime.now()),
            data={"type": "genesis"},
            previous_hash="0"
        )
        self.chain.append(genesis_block)
        self.save_chain()

    def get_last_block(self):
        return self.chain[-1]

    def add_block(self, data):
        last_block = self.get_last_block()

        new_block = Block(
            index=last_block.index + 1,
            timestamp=str(datetime.now()),
            data=data,
            previous_hash=last_block.hash
        )

        if not self.has_block(new_block.hash):
            self.chain.append(new_block)
            self.save_chain()

        return new_block

    def is_valid_block(self, new_block, last_block):
        if new_block.previous_hash != last_block.hash:
            return False

        if new_block.calculate_hash() != new_block.hash:
            return False

        return True

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            if current.hash != current.calculate_hash():
                return False

            if current.previous_hash != previous.hash:
                return False

        return True

    def add_block_remote(self, block_dict):

        new_block = Block(
           index=block_dict["index"],
           timestamp=block_dict["timestamp"],
           data=block_dict["data"],
           previous_hash=block_dict["previous_hash"]
        )

        if new_block.calculate_hash() != block_dict["hash"]:
           print("❌ Invalid hash")
           return

        # 🔥 IMPORTANT: don't care about index or previous_hash across nodes
        self.chain.append(new_block)
        self.save_chain()

        print("🟢 Block merged successfully from remote node")
  


    def has_block(self, block_hash):
        for block in self.chain:
            if block.hash == block_hash:
                return True
        return False

    def save_chain(self):
        chain_data = []

        for block in self.chain:
            chain_data.append({
                "index": block.index,
                "timestamp": block.timestamp,
                "data": block.data,
                "previous_hash": block.previous_hash,
                "hash": block.hash
            })

        with open(self.file_path, "w") as f:
            json.dump(chain_data, f, indent=4)

    def load_chain(self):
        try:
            with open(self.file_path, "r") as f:
                chain_data = json.load(f)

                for blk in chain_data:
                    block = Block(
                        index=blk["index"],
                        timestamp=blk["timestamp"],
                        data=blk["data"],
                        previous_hash=blk["previous_hash"]
                    )
                    block.hash = blk["hash"]
                    self.chain.append(block)

        except FileNotFoundError:
            self.chain = []