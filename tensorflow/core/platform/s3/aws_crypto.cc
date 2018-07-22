/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/platform/s3/aws_crypto.h"
#include <openssl/cipher.h>
#include <openssl/err.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

#include <aws/core/utils/crypto/HashResult.h>
#include <aws/core/utils/logging/LogMacros.h>
#include <aws/s3/S3Client.h>

namespace tensorflow {

class AWSSha256HMACOpenSSLImpl : public Aws::Utils::Crypto::HMAC {
 public:
  AWSSha256HMACOpenSSLImpl() {}

  virtual ~AWSSha256HMACOpenSSLImpl() = default;

  virtual Aws::Utils::Crypto::HashResult Calculate(
      const Aws::Utils::ByteBuffer& toSign,
      const Aws::Utils::ByteBuffer& secret) override {
    unsigned int length = SHA256_DIGEST_LENGTH;
    Aws::Utils::ByteBuffer digest(length);
    memset(digest.GetUnderlyingData(), 0, length);

    HMAC_CTX ctx;
    HMAC_CTX_init(&ctx);

    HMAC_Init_ex(&ctx, secret.GetUnderlyingData(),
    static_cast<int>(secret.GetLength()), EVP_sha256(), NULL);
    HMAC_Update(&ctx, toSign.GetUnderlyingData(), toSign.GetLength());
    HMAC_Final(&ctx, digest.GetUnderlyingData(), &length);
    HMAC_CTX_cleanup(&ctx);

    return Aws::Utils::Crypto::HashResult(std::move(digest));
  }
};

class AWSSecureRandomBytesOpenSSLImpl : public Aws::Utils::Crypto::SecureRandomBytes {
public:
  AWSSecureRandomBytesOpenSSLImpl() {}

  ~AWSSecureRandomBytesOpenSSLImpl() = default;

  virtual void GetBytes(unsigned char* buffer, size_t bufferSize) override {
    int success = RAND_bytes(buffer, static_cast<int>(bufferSize));
    if (success != 1) {
        m_failure = true;
    }
  }
};

class AWSSha256OpenSSLImpl : public Aws::Utils::Crypto::Hash {
 public:
  AWSSha256OpenSSLImpl() {}

  virtual ~AWSSha256OpenSSLImpl() = default;

  virtual Aws::Utils::Crypto::HashResult Calculate(
      const Aws::String& str) override {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.data(), str.size());

    Aws::Utils::ByteBuffer hash(SHA256_DIGEST_LENGTH);
    SHA256_Final(hash.GetUnderlyingData(), &sha256);

    return Aws::Utils::Crypto::HashResult(std::move(hash));
  }

  virtual Aws::Utils::Crypto::HashResult Calculate(
      Aws::IStream& stream) override {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);

    auto currentPos = stream.tellg();
    if (currentPos == std::streampos(std::streamoff(-1))) {
      currentPos = 0;
      stream.clear();
    }

    stream.seekg(0, stream.beg);

    char streamBuffer
        [Aws::Utils::Crypto::Hash::INTERNAL_HASH_STREAM_BUFFER_SIZE];
    while (stream.good()) {
      stream.read(streamBuffer,
     Aws::Utils::Crypto::Hash::INTERNAL_HASH_STREAM_BUFFER_SIZE);
      auto bytesRead = stream.gcount();

      if (bytesRead > 0) {
        SHA256_Update(&sha256, streamBuffer, static_cast<size_t>(bytesRead));
      }
    }

    stream.clear();
    stream.seekg(currentPos, stream.beg);

    Aws::Utils::ByteBuffer hash(SHA256_DIGEST_LENGTH);
    SHA256_Final(hash.GetUnderlyingData(), &sha256);

    return Aws::Utils::Crypto::HashResult(std::move(hash));
  }
};

//class AWSAESGCMCipherOpenSSL : public Aws::Utils::Crypto::SymmetricCipher  {
class OpenSSLCipher : public Aws::Utils::Crypto::SymmetricCipher  {
public:
                OpenSSLCipher(const Aws::Utils::CryptoBuffer& key, size_t ivSize, bool ctrMode = false);
                OpenSSLCipher(Aws::Utils::CryptoBuffer&& key, Aws::Utils::CryptoBuffer&& initializationVector,
                              Aws::Utils::CryptoBuffer&& tag = Aws::Utils::CryptoBuffer(0));
                OpenSSLCipher(const Aws::Utils::CryptoBuffer& key, const Aws::Utils::CryptoBuffer& initializationVector,
                              const Aws::Utils::CryptoBuffer& tag = Aws::Utils::CryptoBuffer(0));

                OpenSSLCipher(const OpenSSLCipher& other) = delete;

                OpenSSLCipher& operator=(const OpenSSLCipher& other) = delete;

                OpenSSLCipher(OpenSSLCipher&& toMove);

                OpenSSLCipher& operator=(OpenSSLCipher&& toMove) = default;


                virtual ~OpenSSLCipher();

                Aws::Utils::CryptoBuffer EncryptBuffer(const Aws::Utils::CryptoBuffer& unEncryptedData) override;

                Aws::Utils::CryptoBuffer FinalizeEncryption() override;
                Aws::Utils::CryptoBuffer DecryptBuffer(const Aws::Utils::CryptoBuffer& encryptedData) override;

                Aws::Utils::CryptoBuffer FinalizeDecryption() override;

                void Reset() override;

            protected:
                virtual size_t GetBlockSizeBytes() const = 0;

                virtual size_t GetKeyLengthBits() const = 0;

                EVP_CIPHER_CTX* m_encryptor_ctx;
                EVP_CIPHER_CTX* m_decryptor_ctx;

            private:
                void Init();
                void Cleanup();
};
            static const char* OPENSSL_LOG_TAG = "OpenSSLCipher";

            void LogErrors(const char* logTag = OPENSSL_LOG_TAG)
            {
                unsigned long errorCode = ERR_get_error();
                char errStr[256];
                ERR_error_string_n(errorCode, errStr, 256);

                AWS_LOGSTREAM_ERROR(logTag, errStr);
            }

            OpenSSLCipher::OpenSSLCipher(const Aws::Utils::CryptoBuffer& key, size_t blockSizeBytes, bool ctrMode) :
                    SymmetricCipher(key, blockSizeBytes, ctrMode), m_encryptor_ctx(nullptr), m_decryptor_ctx(nullptr)
            {
                Init();
            }

            OpenSSLCipher::OpenSSLCipher(OpenSSLCipher&& toMove) : SymmetricCipher(std::move(toMove)),
                    m_encryptor_ctx(nullptr), m_decryptor_ctx(nullptr)
            {
                Init();
                EVP_CIPHER_CTX_copy(m_encryptor_ctx, toMove.m_encryptor_ctx);
                EVP_CIPHER_CTX_copy(m_decryptor_ctx, toMove.m_decryptor_ctx);
                EVP_CIPHER_CTX_cleanup(toMove.m_encryptor_ctx);
                EVP_CIPHER_CTX_cleanup(toMove.m_decryptor_ctx);
            }

            OpenSSLCipher::OpenSSLCipher(Aws::Utils::CryptoBuffer&& key, Aws::Utils::CryptoBuffer&& initializationVector, Aws::Utils::CryptoBuffer&& tag) :
                    SymmetricCipher(std::move(key), std::move(initializationVector), std::move(tag)),
                    m_encryptor_ctx(nullptr), m_decryptor_ctx(nullptr)
            {
                Init();
            }

            OpenSSLCipher::OpenSSLCipher(const Aws::Utils::CryptoBuffer& key, const Aws::Utils::CryptoBuffer& initializationVector,
                                         const Aws::Utils::CryptoBuffer& tag) :
                    SymmetricCipher(key, initializationVector, tag), m_encryptor_ctx(nullptr), m_decryptor_ctx(nullptr)
            {
                Init();
            }

            OpenSSLCipher::~OpenSSLCipher()
            {
                Cleanup();
                if (m_encryptor_ctx)
                {
                    EVP_CIPHER_CTX_free(m_encryptor_ctx);
                    m_encryptor_ctx = nullptr;
                }
                if (m_decryptor_ctx)
                {
                    EVP_CIPHER_CTX_free(m_decryptor_ctx);
                    m_decryptor_ctx = nullptr;
                }
            }

            void OpenSSLCipher::Init()
            {
                if (!m_encryptor_ctx)
                {
                    // EVP_CIPHER_CTX_init() will be called inside EVP_CIPHER_CTX_new().
                    m_encryptor_ctx = EVP_CIPHER_CTX_new();
                    assert(m_encryptor_ctx != nullptr);
                }
                else
                {   // _init is the same as _reset after openssl 1.1
                    EVP_CIPHER_CTX_init(m_encryptor_ctx);
                }
                if (!m_decryptor_ctx)
                {
                    // EVP_CIPHER_CTX_init() will be called inside EVP_CIPHER_CTX_new().
                    m_decryptor_ctx = EVP_CIPHER_CTX_new();
                    assert(m_decryptor_ctx != nullptr);
                }
                else
                {   // _init is the same as _reset after openssl 1.1
                    EVP_CIPHER_CTX_init(m_decryptor_ctx);
                }
            }

            Aws::Utils::CryptoBuffer OpenSSLCipher::EncryptBuffer(const Aws::Utils::CryptoBuffer& unEncryptedData)
            {
                if (m_failure)
                {
                    AWS_LOGSTREAM_FATAL(OPENSSL_LOG_TAG, "Cipher not properly initialized for encryption. Aborting");
                    return Aws::Utils::CryptoBuffer();
                }

                int lengthWritten = static_cast<int>(unEncryptedData.GetLength() + (GetBlockSizeBytes() - 1));
                Aws::Utils::CryptoBuffer encryptedText(static_cast<size_t>( lengthWritten + (GetBlockSizeBytes() - 1)));

                if (!EVP_EncryptUpdate(m_encryptor_ctx, encryptedText.GetUnderlyingData(), &lengthWritten,
                                       unEncryptedData.GetUnderlyingData(),
                                       static_cast<int>(unEncryptedData.GetLength())))
                {
                    m_failure = true;
                    LogErrors();
                    return Aws::Utils::CryptoBuffer();
                }

                if (static_cast<size_t>(lengthWritten) < encryptedText.GetLength())
                {
                    return Aws::Utils::CryptoBuffer(encryptedText.GetUnderlyingData(), static_cast<size_t>(lengthWritten));
                }

                return encryptedText;
            }

            Aws::Utils::CryptoBuffer OpenSSLCipher::FinalizeEncryption()
            {
                if (m_failure)
                {
                    AWS_LOGSTREAM_FATAL(OPENSSL_LOG_TAG,
                                        "Cipher not properly initialized for encryption finalization. Aborting");
                    return Aws::Utils::CryptoBuffer();
                }

                Aws::Utils::CryptoBuffer finalBlock(GetBlockSizeBytes());
                int writtenSize = 0;
                if (!EVP_EncryptFinal_ex(m_encryptor_ctx, finalBlock.GetUnderlyingData(), &writtenSize))
                {
                    m_failure = true;
                    LogErrors();
                    return Aws::Utils::CryptoBuffer();
                }
                return Aws::Utils::CryptoBuffer(finalBlock.GetUnderlyingData(), static_cast<size_t>(writtenSize));
            }

            Aws::Utils::CryptoBuffer OpenSSLCipher::DecryptBuffer(const Aws::Utils::CryptoBuffer& encryptedData)
            {
                if (m_failure)
                {
                    AWS_LOGSTREAM_FATAL(OPENSSL_LOG_TAG, "Cipher not properly initialized for decryption. Aborting");
                    return Aws::Utils::CryptoBuffer();
                }

                int lengthWritten = static_cast<int>(encryptedData.GetLength() + (GetBlockSizeBytes() - 1));
                Aws::Utils::CryptoBuffer decryptedText(static_cast<size_t>(lengthWritten));

                if (!EVP_DecryptUpdate(m_decryptor_ctx, decryptedText.GetUnderlyingData(), &lengthWritten,
                                       encryptedData.GetUnderlyingData(),
                                       static_cast<int>(encryptedData.GetLength())))
                {
                    m_failure = true;
                    LogErrors();
                    return Aws::Utils::CryptoBuffer();
                }

                if (static_cast<size_t>(lengthWritten) < decryptedText.GetLength())
                {
                    return Aws::Utils::CryptoBuffer(decryptedText.GetUnderlyingData(), static_cast<size_t>(lengthWritten));
                }

                return decryptedText;
            }

            Aws::Utils::CryptoBuffer OpenSSLCipher::FinalizeDecryption()
            {
                if (m_failure)
                {
                    AWS_LOGSTREAM_FATAL(OPENSSL_LOG_TAG,
                                        "Cipher not properly initialized for decryption finalization. Aborting");
                    return Aws::Utils::CryptoBuffer();
                }

                Aws::Utils::CryptoBuffer finalBlock(GetBlockSizeBytes());
                int writtenSize = static_cast<int>(finalBlock.GetLength());
                if (!EVP_DecryptFinal_ex(m_decryptor_ctx, finalBlock.GetUnderlyingData(), &writtenSize))
                {
                    m_failure = true;
                    LogErrors();
                    return Aws::Utils::CryptoBuffer();
                }
                return Aws::Utils::CryptoBuffer(finalBlock.GetUnderlyingData(), static_cast<size_t>(writtenSize));
            }

            void OpenSSLCipher::Reset()
            {
                Cleanup();
                Init();
            }

            void OpenSSLCipher::Cleanup()
            {
                m_failure = false;

                EVP_CIPHER_CTX_cleanup(m_encryptor_ctx);
                EVP_CIPHER_CTX_cleanup(m_decryptor_ctx);
            }
            class AESGCMCipherOpenSSL : public OpenSSLCipher
            {
            public:
                AESGCMCipherOpenSSL(const Aws::Utils::CryptoBuffer& key);

                AESGCMCipherOpenSSL(Aws::Utils::CryptoBuffer&& key, Aws::Utils::CryptoBuffer&& initializationVector,
                                       Aws::Utils::CryptoBuffer&& tag = Aws::Utils::CryptoBuffer(0));

                AESGCMCipherOpenSSL(const Aws::Utils::CryptoBuffer& key, const Aws::Utils::CryptoBuffer& initializationVector,
                                       const Aws::Utils::CryptoBuffer& tag = Aws::Utils::CryptoBuffer(0));

                AESGCMCipherOpenSSL(const AESGCMCipherOpenSSL& other) = delete;

                AESGCMCipherOpenSSL& operator=(const AESGCMCipherOpenSSL& other) = delete;

                AESGCMCipherOpenSSL(AESGCMCipherOpenSSL&& toMove) = default;

                Aws::Utils::CryptoBuffer FinalizeEncryption() override;

            protected:
                size_t GetBlockSizeBytes() const override;

                size_t GetKeyLengthBits() const override;

                size_t GetTagLengthBytes() const;

            private:
                void InitCipher();

                static size_t BlockSizeBytes;
                static size_t IVLengthBytes;
                static size_t KeyLengthBits;
                static size_t TagLengthBytes;
            };

            size_t AESGCMCipherOpenSSL::BlockSizeBytes = 16;
            size_t AESGCMCipherOpenSSL::KeyLengthBits = 256;
            size_t AESGCMCipherOpenSSL::IVLengthBytes = 12;
            size_t AESGCMCipherOpenSSL::TagLengthBytes = 16;

            static const char* GCM_LOG_TAG = "AESGCMCipherOpenSSL";

            AESGCMCipherOpenSSL::AESGCMCipherOpenSSL(const Aws::Utils::CryptoBuffer& key) : OpenSSLCipher(key, IVLengthBytes)
            {
                InitCipher();
            }

            AESGCMCipherOpenSSL::AESGCMCipherOpenSSL(Aws::Utils::CryptoBuffer&& key, Aws::Utils::CryptoBuffer&& initializationVector,
                                                           Aws::Utils::CryptoBuffer&& tag) :
                    OpenSSLCipher(std::move(key), std::move(initializationVector), std::move(tag))
            {
                InitCipher();
            }

            AESGCMCipherOpenSSL::AESGCMCipherOpenSSL(const Aws::Utils::CryptoBuffer& key,
                                                           const Aws::Utils::CryptoBuffer& initializationVector,
                                                           const Aws::Utils::CryptoBuffer& tag) :
                    OpenSSLCipher(key, initializationVector, tag)
            {
                InitCipher();
            }

            Aws::Utils::CryptoBuffer AESGCMCipherOpenSSL::FinalizeEncryption()
            {
                Aws::Utils::CryptoBuffer const& finalBuffer = OpenSSLCipher::FinalizeEncryption();
                m_tag = Aws::Utils::CryptoBuffer(TagLengthBytes);
                if (!EVP_CIPHER_CTX_ctrl(m_encryptor_ctx, EVP_CTRL_GCM_GET_TAG, static_cast<int>(m_tag.GetLength()),
                //if (!EVP_CIPHER_CTX_ctrl(m_encryptor_ctx, EVP_CTRL_CCM_GET_TAG, static_cast<int>(m_tag.GetLength()),
                                         m_tag.GetUnderlyingData()))
                {
                    m_failure = true;
                    LogErrors(GCM_LOG_TAG);
                    return Aws::Utils::CryptoBuffer();
                }

                return finalBuffer;
            }

            void AESGCMCipherOpenSSL::InitCipher()
            {
                if (!(EVP_EncryptInit_ex(m_encryptor_ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) &&
                        EVP_EncryptInit_ex(m_encryptor_ctx, nullptr, nullptr, m_key.GetUnderlyingData(),
                                           m_initializationVector.GetUnderlyingData()) &&
                        EVP_CIPHER_CTX_set_padding(m_encryptor_ctx, 0)) ||
                    !(EVP_DecryptInit_ex(m_decryptor_ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) &&
                        EVP_DecryptInit_ex(m_decryptor_ctx, nullptr, nullptr, m_key.GetUnderlyingData(),
                                           m_initializationVector.GetUnderlyingData()) &&
                        EVP_CIPHER_CTX_set_padding(m_decryptor_ctx, 0)))
                {
                    m_failure = true;
                    LogErrors(GCM_LOG_TAG);
                    return;
                }

                //tag should always be set in GCM decrypt mode
                if (m_tag.GetLength() > 0)
                {
                    if (m_tag.GetLength() < TagLengthBytes)
                    {
                        AWS_LOGSTREAM_ERROR(GCM_LOG_TAG,
                                            "Illegal attempt to decrypt an AES GCM payload without a valid tag set: tag length=" <<
                                                    m_tag.GetLength());
                        m_failure = true;
                        return;
                    }

                    if (!EVP_CIPHER_CTX_ctrl(m_decryptor_ctx, EVP_CTRL_GCM_SET_TAG, static_cast<int>(m_tag.GetLength()),
                                             m_tag.GetUnderlyingData()))
                    {
                        m_failure = true;
                        LogErrors(GCM_LOG_TAG);
                    }
                }
            }

            size_t AESGCMCipherOpenSSL::GetBlockSizeBytes() const
            {
                return BlockSizeBytes;
            }

            size_t AESGCMCipherOpenSSL::GetKeyLengthBits() const
            {
                return KeyLengthBits;
            }

            size_t AESGCMCipherOpenSSL::GetTagLengthBytes() const
            {
                return TagLengthBytes;
            }



std::shared_ptr<Aws::Utils::Crypto::Hash>
AWSSHA256Factory::CreateImplementation() const {
  return Aws::MakeShared<AWSSha256OpenSSLImpl>(AWSCryptoAllocationTag);
}

std::shared_ptr<Aws::Utils::Crypto::HMAC>
AWSSHA256HmacFactory::CreateImplementation() const {
  return Aws::MakeShared<AWSSha256HMACOpenSSLImpl>(AWSCryptoAllocationTag);
}

std::shared_ptr<Aws::Utils::Crypto::SecureRandomBytes>
AWSSecureRandomFactory::CreateImplementation() const {
  return Aws::MakeShared<AWSSecureRandomBytesOpenSSLImpl>(AWSCryptoAllocationTag);
}

std::shared_ptr<Aws::Utils::Crypto::SymmetricCipher>
AWSAESGCMFactory::CreateImplementation(const Aws::Utils::CryptoBuffer& key) const {
  return Aws::MakeShared<AESGCMCipherOpenSSL>(AWSCryptoAllocationTag, key);
}

std::shared_ptr<Aws::Utils::Crypto::SymmetricCipher>
AWSAESGCMFactory::CreateImplementation(const Aws::Utils::CryptoBuffer& key, const Aws::Utils::CryptoBuffer& iv, const Aws::Utils::CryptoBuffer& tag) const {
  return Aws::MakeShared<AESGCMCipherOpenSSL>(AWSCryptoAllocationTag, key, iv, tag);
}

std::shared_ptr<Aws::Utils::Crypto::SymmetricCipher>
AWSAESGCMFactory::CreateImplementation(Aws::Utils::CryptoBuffer&& key, Aws::Utils::CryptoBuffer&& iv, Aws::Utils::CryptoBuffer&& tag) const {
  return Aws::MakeShared<AESGCMCipherOpenSSL>(AWSCryptoAllocationTag, std::move(key), std::move(iv), std::move(tag));
}

}  // namespace tensorflow
