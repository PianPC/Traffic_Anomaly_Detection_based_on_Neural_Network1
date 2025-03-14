from scapy.layers.tls.record import TLS
from scapy.layers.http import HTTP

class ProtocolAnalyzer:
    @staticmethod
    def detect_encrypted(pkt):
        """识别加密协议"""
        meta = {
            "is_encrypted": False,
            "protocol": "Unknown",
            "sni": None,
            "tls_version": None
        }
        
        # TLS/HTTPS检测
        if pkt.haslayer(TLS):
            meta.update({
                "is_encrypted": True,
                "protocol": "TLS",
                "tls_version": pkt[TLS].version
            })
            # 提取SNI
            if hasattr(pkt[TLS], 'sni'):
                meta["sni"] = pkt[TLS].sni.decode()
            return meta
                
        # HTTP/2 over TLS检测
        if pkt.haslayer(HTTP) and pkt[TCP].dport == 443:
            meta.update({
                "is_encrypted": True,
                "protocol": "HTTPS"
            })
            return meta
        
        # QUIC检测（基于UDP的快速协议）
        if pkt.haslayer(UDP) and len(pkt[UDP].payload) > 0:
            quic_flags = pkt[UDP].payload.load[0]
            if (quic_flags & 0x40) != 0:  # QUIC公共头标志位
                meta.update({
                    "is_encrypted": True,
                    "protocol": "QUIC"
                })
                return meta
        
        return meta