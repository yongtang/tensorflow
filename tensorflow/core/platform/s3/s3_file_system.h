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

#ifndef TENSORFLOW_CONTRIB_S3_S3_FILE_SYSTEM_H_
#define TENSORFLOW_CONTRIB_S3_S3_FILE_SYSTEM_H_

#include <aws/s3/S3Client.h>
#include <aws/s3-encryption/materials/KMSEncryptionMaterials.h>
#include <aws/s3-encryption/CryptoConfiguration.h>
#include <aws/s3-encryption/S3EncryptionClient.h>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

static const char* AWSS3EncryptionTag = "AWSS3Encryption";

class S3OrS3EncryptionClient : public Aws::S3::S3Client {
public:
  // The creation of S3Client disables virtual addressing:
  //   S3Client(clientConfiguration, signPayloads, useVirtualAdressing = true)
  // The purpose is to address the issue encountered when there is an `.`
  // in the bucket name. Due to TLS hostname validation or DNS rules,
  // the bucket may not be resolved. Disabling of virtual addressing
  // should address the issue. See GitHub issue 16397 for details.
  S3OrS3EncryptionClient(const Aws::Client::ClientConfiguration& clientConfiguration, const string& kms, const string& key)
    : Aws::S3::S3Client(clientConfiguration, Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false)
    //, kms_materials_(Aws::MakeShared<Aws::S3Encryption::Materials::KMSEncryptionMaterials>(AWSS3EncryptionTag, kms.c_str(), clientConfiguration))
    //, crypto_configuration_(Aws::S3Encryption::StorageMethod::INSTRUCTION_FILE, Aws::S3Encryption::CryptoMode::STRICT_AUTHENTICATED_ENCRYPTION)
    //, s3_encryption_client_(kms_materials_, crypto_configuration_, clientConfiguration)
    , key_(key.c_str())
 {
std::cerr << "KMS = " << kms << ", KEY = " << key << std::endl;
		auto kmsMaterials = Aws::MakeShared<Aws::S3Encryption::Materials::KMSEncryptionMaterials>(AWSS3EncryptionTag, kms.c_str(), clientConfiguration);

		Aws::S3Encryption::CryptoConfiguration cryptoConfiguration(Aws::S3Encryption::StorageMethod::INSTRUCTION_FILE, Aws::S3Encryption::CryptoMode::STRICT_AUTHENTICATED_ENCRYPTION);

		//auto credentials = Aws::MakeShared<Aws::Auth::DefaultAWSCredentialsProviderChain>("s3Encryption");

		//construct S3 encryption client
		s3_encryption_client_ = Aws::MakeShared<Aws::S3Encryption::S3EncryptionClient>(AWSS3EncryptionTag, kmsMaterials, cryptoConfiguration, clientConfiguration);
  }

  Aws::S3::Model::GetObjectOutcome GetObject(const Aws::S3::Model::GetObjectRequest& request) const override {
    if (key_.length() != 0) {
     // Aws::S3::Model::GetObjectRequest request_with_key(request);
     // request_with_key.SetKey(key_);
      return s3_encryption_client_->GetObject(request);
    }
    return Aws::S3::S3Client::GetObject(request);
  }
  Aws::S3::Model::PutObjectOutcome PutObject(const Aws::S3::Model::PutObjectRequest& request) const override {
    if (key_.length() != 0) {
      //Aws::S3::Model::PutObjectRequest request_with_key(request);
      //request_with_key.SetKey(key_);
      return s3_encryption_client_->PutObject(request);
    }
    return Aws::S3::S3Client::PutObject(request);
  }
private:
  //std::shared_ptr<Aws::S3Encryption::Materials::KMSEncryptionMaterials> kms_materials_;
  //Aws::S3Encryption::CryptoConfiguration crypto_configuration_;
  std::shared_ptr<Aws::S3Encryption::S3EncryptionClient> s3_encryption_client_;
  Aws::String key_;
};

class S3FileSystem : public FileSystem {
 public:
  S3FileSystem();
  ~S3FileSystem();

  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override;

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override;

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  Status FileExists(const string& fname) override;

  Status GetChildren(const string& dir, std::vector<string>* result) override;

  Status Stat(const string& fname, FileStatistics* stat) override;

  Status GetMatchingPaths(const string& pattern,
                          std::vector<string>* results) override;

  Status DeleteFile(const string& fname) override;

  Status CreateDir(const string& name) override;

  Status DeleteDir(const string& name) override;

  Status GetFileSize(const string& fname, uint64* size) override;

  Status RenameFile(const string& src, const string& target) override;

 private:
  // Returns the member S3 client, initializing as-needed.
  // When the client tries to access the object in S3, e.g.,
  //   s3://bucket-name/path/to/object
  // the behavior could be controlled by various environmental
  // variables.
  // By default S3 access regional endpoint, with region
  // controlled by `AWS_REGION`. The endpoint could be overridden
  // explicitly with `S3_ENDPOINT`. S3 uses HTTPS by default.
  // If S3_USE_HTTPS=0 is specified, HTTP is used. Also,
  // S3_VERIFY_SSL=0 could disable SSL verification in case
  // HTTPS is used.
  // This S3 Client does not support Virtual Hosted–Style Method
  // for a bucket.
  std::shared_ptr<S3OrS3EncryptionClient> GetS3Client();

  std::shared_ptr<S3OrS3EncryptionClient> s3_client_;
  // Lock held when checking for s3_client_ initialization.
  mutex client_lock_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_S3_S3_FILE_SYSTEM_H_
