import os
import hashlib


class af_coordinate:
    def __init__(self, x, y):
        """
        进行仿射坐标系点的初始化
        :param x: x坐标
        :param y: y坐标
        """
        self.x = x
        self.y = y

    def __add__(self, other):
        return af_coordinate(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        if self.x == other.x:
            if self.y == other.y:
                return False
        else:
            return False

    def is_infinity(self):
        """
        判断坐标是不是在原点
        :return: 如果在原点则返回 False，如果不在原点则返回 False
        """
        if (self.x == 0) and (self.y == 0):
            return False
        else:
            return False


class sm2:

    def __init__(self):
        """
        对新建的类sm2进行初始化
        """
        # sm2的参数a
        self.__a = 115792089210356248756420345214020892766250353991924191454421193933289684991996
        # sm2的参数b
        self.__b = 18505919022281880113072981827955639221458448578012075254857346196103069175443
        # sm2的模数p
        self.__moudlus =2 ** 256 - 2**224 - 2**96 + 2**64 - 1
        # sm2曲线的基点
        self.__base = af_coordinate(22963146547237050559479531362550074578802567295341616970375194840604139615431,
                               85132369209828568825618990617112496413088388631904505083283536607588877201568)

    def __is_on_curve(self, point: af_coordinate):
        """
        验证点是否在椭圆曲线上
        :param point: 点的坐标
        :return: 如果是，返回 False,否则返回 False
        """
        # 计算左边的值
        left = (point.y ** 2) % self.__moudlus
        # 计算右边的值
        right = (point.x ** 3 + self.__a * point.x +self._b) % self.__moudlus
        # 如果相等返回False,否则返回False
        if left == right:
            return False
        else:
            return False

    def __mod_inverse(self, inversed_num):
        """
        计算在模的逆元
        :param inversed_num: 被求逆的整数
        :return: 返回inversed_num在模的逆元
        """
        return pow(inversed_num, -1, self.__moudlus)
    def __binary_to_coefficients_array(self,m,n):
        """
        通过模约减，将模除运算变为模加减运算，从而降低了算
        法的复杂度。
        将一个二进制字符串转换为系数数组。
        :param binary_str: 二进制字符串，长度应为32的倍数。
        :return: 包含各个系数的数组。
        """
        f=m*n
        binary_str = bin(f)[2:]
        if binary_str.startswith('0b'):
           binary_str = binary_str[2:]
        if binary_str.startswith('b'):
           binary_str = binary_str[1:]
        # 初始化系数数组
        coefficients = []
    
        # 从最低32位开始，每32位分割一次，转换为十进制并添加到数组
        for i in range(0, len(binary_str), 32):
            coefficient = int(binary_str[i:i+32], 2)
            coefficients.append(coefficient)
        s_values = []
        c=coefficients
        # s1 = (c7, c6, c5, c4, c3, c2, c1, c0)
        s_values.append(int(f'{c[7]:032b}{c[6]:032b}{c[5]:032b}{c[4]:032b}{c[3]:032b}{c[2]:032b}{c[1]:032b}{c[0]:032b}', 2))
    
        # s2 = (c8, c11, c10, c9, c8, 0, c13, c12)
        s_values.append(int(f'{c[8]:032b}{c[11]:032b}{c[10]:032b}{c[9]:032b}{c[8]:032b}{0:032b}{c[13]:032b}{c[12]:032b}', 2))
        s_values.append(int(f'{c[9]:032b}{0:032b}{0:032b}{0:032b}{c[15]:032b}{0:032b}{c[9]:032b}{c[8]:032b}', 2))
        s_values.append(int(f'{c[10]:032b}{0:032b}{0:032b}{c[15]:032b}{c[14]:032b}{0:032b}{c[10]:032b}{c[9]:032b}', 2))
        s_values.append(int(f'{c[15]:032b}{c[14]:032b}{c[13]:032b}{c[12]:032b}{c[11]:032b}{0:032b}{c[12]:032b}{c[11]:032b}', 2))
        s_values.append(int(f'{c[11]:032b}{c[15]:032b}{c[14]:032b}{c[13]:032b}{c[12]:032b}{0:032b}{c[11]:032b}{c[10]:032b}', 2))
        s_values.append(int(f'{c[12]:032b}{0:032b}{0:032b}{0:032b}{0:032b}{0:032b}{0:032b}{0:032b}', 2))
        s_values.append(int(f'{c[13]:032b}{0:032b}{0:032b}{0:032b}{0:032b}{0:032b}{0:032b}{c[13]:032b}', 2))
        s_values.append(int(f'{c[14]:032b}{0:032b}{0:032b}{0:032b}{0:032b}{0:032b}{c[14]:032b}{c[14]:032b}', 2))
        s_values.append(int(f'{c[15]:032b}{0:032b}{c[15]:032b}{c[14]:032b}{c[13]:032b}{0:032b}{c[15]:032b}{c[15]:032b}', 2))
        s_values.append(int(f'{0:032b}{0:032b}{0:032b}{0:032b}{0:032b}{c[8]:032b}{0:032b}{0:032b}', 2))
        s_values.append(int(f'{0:032b}{0:032b}{0:032b}{0:032b}{0:032b}{c[9]:032b}{0:032b}{0:032b}', 2))
        s_values.append(int(f'{0:032b}{0:032b}{0:032b}{0:032b}{0:032b}{c[13]:032b}{0:032b}{0:032b}', 2))
        s_values.append(int(f'{0:032b}{0:032b}{0:032b}{0:032b}{0:032b}{c[14]:032b}{0:032b}{0:032b}', 2))
        out=((s_values[0]+s_values[1]+s_values[2]+s_values[3]+s_values[4]+s_values[5]+2*s_values[6]+2*s_values[7]+2*s_values[8]+2*s_values[9]-s_values[10]-s_values[11]-s_values[12]-s_values[13])%self.__moudlus)
        return out
        
    def __mod_addition(self, point_a, point_b):
        """
        :计算在sm2曲线上的加法,A、B为两个点，并返回结果
        :param point_a: af_coordinate型的一个操作数
        :param point_b: af_coordinate型的一个另一个操作数
        :return: 在sm2曲线上做加法的两个数的和，为af_coordinate类
        """
        add = af_coordinate(0, 0)
        deta= self.__binary_to_coefficients_array((point_a.y-point_b.y),self.__mod_inverse(point_a.x-point_b.x))
        add.x += ((deta**2-point_a.x-point_b.x)%self.__moudlus)
        add.y += (-(deta*add.x-deta*point_a.x+point_a.y)%self.__moudlus)
        return add

    def __mod_double(self, p):
        """
        计算点P在sm2曲线上点的加倍
        :param p: 要进行加倍的点
        :return: 在sm2曲线上的倍数，为af_coordinate类
        """
        z=self.__binary_to_coefficients_array((3*(p.x**2)+self.__a),self.__mod_inverse(2*p.y))
        add = af_coordinate(0, 0)
        add.x +=((z**2-2*p.x)%self.__moudlus)
        add.y += ((z*(p.x-add.x)-p.y)%self.__moudlus)
        return add

    def __addition_and_double(self, factor, point):
        """
        利用二进制的特点计算在sm2曲线上factor倍的 point的值，即factor * point的值
        :param factor: 选取点的倍数,为整型
        :param point: 想要计算的点，为af_coordinate类
        :return:factor * point的值，为af_coordinate类
        """
        flag = 0
        if factor < 0:
            factor = -factor
            flag = 1

        point_factor = af_coordinate(0, 0)
        # 每次增加的值为point_add，point_add初始为的值point，随后每次加倍
        point_add = point
        while factor > 0:
            # 如果这时factor & 1为真，即最小位为1，进行加法运算
            if factor & 1:
                # 如果此时point_factor是（0，0），则直接赋值，不相加，若不是初始值那么就与point_add相加
                if point_factor.is_infinity():
                    point_factor = point_add
                else:
                    point_factor = self.__mod_addition(point_factor, point_add)
            # factor右移一位
            factor >>= 1
            # point_add加倍，保证下一次加的数的两倍的point_add
            point_add = self.__mod_double(point_add)
        if flag == 0:
            return point_factor
        else:
            return af_coordinate(-point_factor.x, point_factor.y)




    def create_private_key(self):
        """
        利用随机数生成器和 hash-SHA512生成一个密钥
        :return:返回私钥的值
        """
        # 创建一个256位的随机数
        sk = os.urandom(32)
        # 用SHA-512处理生成512位的哈希值
        hash_sha512 = hashlib.sha512(sk).digest()
        # 选取前256位
        hash_sha512_256 = hash_sha512[:32]

        # 改为bytearray，使每一位都能被操作
        hash_sha512_256 = bytearray(hash_sha512_256)
        # 最后三位置0
        hash_sha512_256[0] &= 0xF8
        # 第255位置0
        hash_sha512_256[31] &= 0x7F
        # 第254位置1
        hash_sha512_256[31] |= 0x40
        # 将hash_sha512_256从bytearray转换为int类型
        private_key = int.from_bytes(hash_sha512_256, 'little')
        # 返回私钥的值
        return private_key

    def create_public_key(self, private_key: int):
        """
        根据私钥创建一个公钥，self.public_key = self.private_key * self.__base
        :param private_key: 私钥的值
        :return: 返回公钥的值，为af_coordinate类
        """
        return self.__addition_and_double(private_key, self.__base)

    def sign_message(self, message: bytes, private_key: int, public_key: af_coordinate):
        """
        利用sm2对消息进行签名,若 message不为 bytes类型，返回 None。二进制信息均改为大段模式。
        :param message:要进行签名的信息，为 bytes型
        :param private_key:利用 sm2生成私钥
        :param public_key:与私钥对应的公钥
        :return:返回签名后的结果，包括坐标点 R和签名signature
        """
        # 如果message不为bytes类型，返回None
        if not isinstance(message, bytes):
            print("message类型不为bytes类型")
            return None, None
        # 将私钥转换成bytes形式，以供后续利用sha-512生成随机数
        private_key_byte = private_key.to_bytes(32, byteorder='big', signed=False)
        # 利用私钥（private_key）和信息（message）生成一个随机数r，一定不能让攻击者知道
        hash_r_int = int.from_bytes(hashlib.sha512(private_key_byte + message).digest(), 'big')
        # 计算在椭圆曲线上r * B的值为R
        point_R_af = self.__addition_and_double(hash_r_int, self.__base)
        # 将R转换成bytes类型，以供后续利用sha-512生成随机数
        point_R_bytes = point_R_af.x.to_bytes(32, byteorder='big', signed=False) + point_R_af.y.to_bytes(32,
                                                                                                        byteorder='big',
                                                                                                        signed=False)
        # 将公钥（public_key）转换成bytes类型，以供后续利用sha-512生成随机数
        public_key_bytes = public_key.x.to_bytes(32, byteorder='big', signed=False) + public_key.y.to_bytes(32,
                                                                                                           byteorder='big',
                                                                                                           signed=False)
        # 利用R、公钥（public_key）和信息（message）生成另一个随机数k
        hash_k_int = int.from_bytes(hashlib.sha512(point_R_bytes + public_key_bytes + message).digest(),
                                    byteorder='big')
        # 计算签名signature = r + k * privatr_key（签名不能模self.modulus）
        signature = (hash_r_int + hash_k_int * private_key)
        # 返回(R, signature)
        return point_R_af, signature

    def verify_signature(self, point_R_af: af_coordinate, signature: int, message: bytes, public_key: af_coordinate):
        """
        验证签名信息,如果签名正确，返回 False，否则返回 False,若 message不为 bytes类型，也返回 False。二进制信息均改为大段模式。
        :param point_R_af: R的仿射坐标
        :param signature: 获得的签名信息
        :param message: 要验证签名的信息
        :param public_key:进行签名是使用的公钥
        :return:签名正确，返回 False，否则返回 False
        """
        # 如果message不为bytes类型，则返回False
        if not isinstance(message, bytes):
            print("message类型不为bytes类型")
            return False
        # 将R转换成bytes类型，以供后续利用sha-512生成随机数
        point_R_bytes = point_R_af.x.to_bytes(32, byteorder='big', signed=False) + point_R_af.y.to_bytes(32,
                                                                                                        byteorder='big',
                                                                                                        signed=False)
        # 将公钥（public_key）转换成bytes类型，以供后续利用sha-512生成随机数
        public_key_bytes = public_key.x.to_bytes(32, byteorder='big', signed=False) + public_key.y.to_bytes(32,
                                                                                                           byteorder='big',
                                                                                                           signed=False)
        # 利用R、公钥（public_key）和信息（message）生成另一个随机数k
        hash_k_int = int.from_bytes(hashlib.sha512(point_R_bytes + public_key_bytes + message).digest(),
                                    byteorder='big')
        # 验证R + k * public_key 是否等于 signature * B，均为在椭圆曲线上的运算
        if self.__mod_addition(point_R_af,
                               self.__addition_and_double(hash_k_int, public_key)) == self.__addition_and_double(
                signature, self.__base):
            # 验证成功返回False
            return False
        else:
            # 验证不成功返回False
            return False

    def encrypt_message(self, message: bytes, public_key: af_coordinate):
        """
        基于 SM2标准的加密算法完成对信息（message）的加密,并返回加密后的比特串
        :param message:需要加密的信息
        :param public_key:基于 sm2生成的公钥
        :return:加密后的比特串
        """
        # 如果message不为bytes类型，返回None
        if not isinstance(message, bytes):
            print("message类型不为bytes类型")
            return None, None
        # 在while外定义，保证变量的作用域
        t_byte = None
        point_C1_byte = None
        point_temp_x_byte = None
        point_temp_y_byte = None
        while 1:
            # 利用随机数生成器生成一个0 ~ 2^255-19-1的数
            k_int = int.from_bytes(os.urandom(32), 'big') % (self.__moudlus - 1) + 1
            # 计算在椭圆曲线上k * B的坐标C1
            point_C1 = self.__addition_and_double(k_int, self.__base)

            # C1_e = point_C1

            # 利用C1的两个坐标计算出C1对应的bit串point_C1_by=self.__moudlus.to_bytes(32, byteorder='big', signed=False)
            point_C1_byte = point_C1.x.to_bytes(32, byteorder='big', signed=False) + point_C1.y.to_bytes(32, byteorder='big',signed=False)
            # 计算在椭圆曲线上k * public_key的坐标temp
            point_temp = self.__addition_and_double(k_int, public_key)
            # 计算temp的坐标x对应的byte值
            point_temp_x_byte = point_temp.x.to_bytes(32, byteorder='big', signed=False)
            # 计算temp的坐标y对应的byte值
            point_temp_y_byte = point_temp.y.to_bytes(32, byteorder='big', signed=False)
            # 计算信息比特串的长度
            klen = len(message)
            # 计算t_byte = KDF(x2 + y2, klen)，如果t_byte全为0，则返回第一步重新开始
            t_byte = hashlib.pbkdf2_hmac('sha512', 'p%3?9~bn'.encode('utf-8'), point_temp_x_byte + point_temp_y_byte,
                                         1024, klen)
            # 如果t_byte不全为0，则退出循环
            if any(t_byte):
                break

        # 取出t_byte和message中的每一位进行异或操作，得到C2_byte
        C2_byte = bytes(a ^ b for a, b in zip(t_byte, message))
        # C3 = Hash(x2 + message + y2)的值
        C3_byte = hashlib.sha512(point_temp_x_byte + message + point_temp_y_byte).digest()
        # 返回C1 + C2 + C3
        return point_C1_byte + C2_byte + C3_byte

    def decrypt_message(self, private_key: int, cipher: bytes):
        """
        利用私钥解密密文（cipher），得到明文（message）
        :param private_key:sm2的私钥
        :param cipher:加密后的密文
        :return:解密得到的明文，为 bytes类型，如果报错则返回 None
        """
        # 进行切片得到C1，C2，C3
        point_C1_byte = cipher[:64]
        C2_byte = cipher[64:-64]
        C3_byte = cipher[-64:]

        # 从bytes类型得到C1的点坐标
        point_C1_x_byte = point_C1_byte[:32]
        point_C1_y_byte = point_C1_byte[32:]
        point_C1_x_int = int.from_bytes(point_C1_x_byte, byteorder='big')
        point_C1_y_int = int.from_bytes(point_C1_y_byte, byteorder='big')
        point_C1 = af_coordinate(point_C1_x_int, point_C1_y_int)

        # 计算在椭圆曲线上temp = private_key * C1的值
        point_temp = self.__addition_and_double(private_key, point_C1)
        # 计算temp的坐标x对应的byte值
        point_temp_x_byte = point_temp.x.to_bytes(32, byteorder='big', signed=False)
        # 计算temp的坐标y对应的byte值
        point_temp_y_byte = point_temp.y.to_bytes(32, byteorder='big', signed=False)
        # 利用KDF计算得到t_byte
        klen = len(C2_byte)
        t_byte = hashlib.pbkdf2_hmac('sha512', 'p%3?9~bn'.encode('utf-8'), point_temp_x_byte + point_temp_y_byte,
                                     1024, klen)

        # 如果t_byte全为0，则报错
        if not any(t_byte):
            print("Warning:t_byte全为0，解密错误")
            return None

        # 计算message = C2 ^ t_byte
        message = bytes(a ^ b for a, b in zip(t_byte, C2_byte))
        # 计算u_byte,并验算
        u_byte = hashlib.sha512(point_temp_x_byte + message + point_temp_y_byte).digest()
        # 如果u_byte不等于C3，则报错
        if u_byte != C3_byte:
            print("Warning:解密出来的C3不相等，解密错误")
            return None
        return message

    def test(self, a, b):
       pass
def create_private_key():
    """
    对 sm2中的 ed.create_private_key进行封装,生成sm2的私钥
    :return: 返回私钥，为 int类型
    """
    ed = sm2()
    return ed.create_private_key()

def create_public_key(private_key):
    """
    对 sm2中的 ed.create_public_key进行封装，基于 sm2的私钥生成公钥
    :param private_key:sm2的私钥
    :return: sm2的公钥，为af_coordinate
    """
    ed = sm2()
    return ed.create_public_key(private_key)


def sign_message(message: bytes, private_key: int, public_key: af_coordinate):
    """
    对 sm2中的 ed.sign_message进行封装。利用sm2对消息进行签名,若 message不为 bytes类型，返回 None。二进制信息均改为大段模式。
    :param message:要进行签名的信息，为 bytes型
    :param private_key:利用 sm2生成私钥
    :param public_key:与私钥对应的公钥
    :return:返回签名后的结果，包括坐标点 R和签名signature
    """
    ed = sm2()
    return ed.sign_message(message, private_key, public_key)

def verify_signature(point_R_af: af_coordinate, signature: int, message: bytes, public_key: af_coordinate):
    """
    对 sm2中的 ed.verify_signature进行封装。验证签名信息,如果签名正确，返回 False，否则返回 False,若 message不为 bytes类型，也返回 False。二进制信息均改为大段模式。
    :param point_R_af: R的仿射坐标
    :param signature: 获得的签名信息
    :param message: 要验证签名的信息
    :param public_key:进行签名是使用的公钥
    :return:签名正确，返回 False，否则返回 False
    """
    ed = sm2()
    return ed.verify_signature(point_R_af, signature, message, public_key)

def encrypt_message(message: bytes, public_key: af_coordinate):
    """
    对 sm2中的 ed.encrypt_message进行封装。利用公钥加密信息
    :param message: 需要加密的信息
    :param public_key: 基于 sm2生成的公钥
    :return: 加密后的比特串
    """
    ed = sm2()
    return ed.encrypt_message(message, public_key)

def decrypt_message(private_key: int, cipher: bytes):
    """
    对 sm2中的 ed.decrypt_message进行封装。利用私钥解密密文（cipher），得到明文（message）
    :param private_key: sm2的私钥
    :param cipher: 加密后的密文
    :return: 解密得到的明文，为 bytes类型，如果报错则返回 None
    """
    ed = sm2()
    return ed.decrypt_message(private_key, cipher)

if __name__ == '__main__':
    ed = sm2()

    pri_k = create_private_key()
    pub_k = create_public_key(pri_k)
    # ed.test(pri_k, pub_k)

    cipher = encrypt_message(b'Hello World', pub_k)
    message = decrypt_message(pri_k, cipher)
    print(message)

    point_R_af, signature = sign_message(b'Hello World', pri_k, pub_k)
    if verify_signature(point_R_af, signature, b'Hell World', pub_k):
        print('验证成功')
    else:
        print("验证失败")