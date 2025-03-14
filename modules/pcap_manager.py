from scapy.utils import PcapWriter, rdpcap

class PcapManager:
    def __init__(self):
        self.writers = {}
    
    def start_capture(self, interface, output_file=None):
        """启动带存储的抓包"""
        if output_file:
            self.writers[interface] = PcapWriter(output_file)
    
    def save_packet(self, pkt, interface):
        """存储单个数据包"""
        if interface in self.writers:
            self.writers[interface].write(pkt)
    
    def load_pcap(self, filepath):
        """加载PCAP文件"""
        return rdpcap(filepath)