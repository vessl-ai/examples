resource "aws_iam_role" "s3_spectrum_role" {
  name = "s3_spectrum_role"

  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "redshift.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
}

data "aws_iam_role" "AWSServiceRoleForRedshift" {
  name = "AWSServiceRoleForRedshift"
}

resource "aws_iam_role_policy_attachment" "s3_read" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
  role       = aws_iam_role.s3_spectrum_role.name
}

resource "aws_iam_role_policy_attachment" "glue_full" {
  policy_arn = "arn:aws:iam::aws:policy/AWSGlueConsoleFullAccess"
  role       = aws_iam_role.s3_spectrum_role.name
}

resource "aws_iam_policy" "s3_full_access_policy" {
  name = "s3_full_access_policy"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": [
        "s3:*"
      ],
      "Effect": "Allow",
      "Resource": "*"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "s3-policy-attachment" {
  role       = aws_iam_role.s3_spectrum_role.name
  policy_arn = aws_iam_policy.s3_full_access_policy.arn
}

resource "aws_redshift_cluster" "feast_redshift_cluster" {
  cluster_identifier = "${var.project_name}-redshift-cluster"
  iam_roles = [
    data.aws_iam_role.AWSServiceRoleForRedshift.arn,
    aws_iam_role.s3_spectrum_role.arn
  ]
  database_name   = var.database_name
  master_username = var.admin_user
  master_password = var.admin_password
  node_type       = var.node_type
  cluster_type    = var.cluster_type
  number_of_nodes = var.nodes

  skip_final_snapshot = true
}

resource "aws_glue_catalog_database" "glue_db" {
  name = var.database_name
}

resource "aws_glue_catalog_table" "zipcode_features_table" {
  name          = "zipcode_features"
  database_name = var.database_name

  table_type = "EXTERNAL_TABLE"

  parameters = {
    EXTERNAL              = "TRUE"
    "parquet.compression" = "SNAPPY"
  }

  storage_descriptor {
    location      = "s3://${var.S3_bucket_name}/source_data/zipcode_features/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      name                  = "my-stream"
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"

      parameters = {
        "serialization.format" = 1
      }
    }

    columns {
      name = "zipcode"
      type = "BIGINT"
    }

    columns {
      name = "city"
      type = "VARCHAR(128)"
    }
    columns {
      name = "state"
      type = "VARCHAR(128)"
    }
    columns {
      name = "location_type"
      type = "VARCHAR(128)"
    }
    columns {
      name = "tax_returns_filed"
      type = "BIGINT"
    }
    columns {
      name = "population"
      type = "BIGINT"
    }
    columns {
      name = "total_wages"
      type = "BIGINT"
    }
    columns {
      name = "event_timestamp"
      type = "timestamp"
    }
    columns {
      name = "created_timestamp"
      type = "timestamp"
    }
  }
  depends_on = [
    resource.aws_glue_catalog_database.glue_db
  ]
}

resource "aws_glue_catalog_table" "credit_history_table" {
  name          = "credit_history"
  database_name = var.database_name

  table_type = "EXTERNAL_TABLE"

  parameters = {
    EXTERNAL              = "TRUE"
    "parquet.compression" = "SNAPPY"
  }

  storage_descriptor {
    location      = "s3://${var.S3_bucket_name}/source_data/credit_history/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      name                  = "my-stream"
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"

      parameters = {
        "serialization.format" = 1
      }
    }
    columns {
      name = "dob_ssn"
      type = "VARCHAR(13)"
    }

    columns {
      name = "credit_card_due"
      type = "BIGINT"
    }
    columns {
      name = "mortgage_due"
      type = "BIGINT"
    }
    columns {
      name = "student_loan_due"
      type = "BIGINT"
    }
    columns {
      name = "vehicle_loan_due"
      type = "BIGINT"
    }
    columns {
      name = "hard_pulls"
      type = "BIGINT"
    }
    columns {
      name = "missed_payments_2y"
      type = "BIGINT"
    }
    columns {
      name = "missed_payments_1y"
      type = "BIGINT"
    }
    columns {
      name = "missed_payments_6m"
      type = "BIGINT"
    }
    columns {
      name = "bankruptcies"
      type = "BIGINT"
    }
    columns {
      name = "event_timestamp"
      type = "timestamp"
    }
    columns {
      name = "created_timestamp"
      type = "timestamp"
    }
  }
  depends_on = [
    resource.aws_glue_catalog_database.glue_db
  ]
}
