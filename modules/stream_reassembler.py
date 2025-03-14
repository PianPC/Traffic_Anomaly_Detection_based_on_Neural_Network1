from collections import defaultdict

class StreamReassembler:
    def __init__(self):
        self.tcp_streams = defaultdict(bytes)
        self.quic_streams = defaultdict(list)
    
    def process_tcp(self, pkt):
        """TCP流重组"""
        if not pkt.haslayer(Raw):
            return None
        
        stream_id = self._get_stream_id(pkt)
        payload = pkt[Raw].load
        
        # 处理乱序和重传（简单实现）
        self.tcp_streams[stream_id] += payload
        return self.tcp_streams[stream_id]
    
    def _get_stream_id(self, pkt):
        return tuple(sorted([(pkt[IP].src, pkt[TCP].sport),
                           (pkt[IP].dst, pkt[TCP].dport)]))