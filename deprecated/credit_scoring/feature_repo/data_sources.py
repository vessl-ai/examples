from feast import RedshiftSource

zipcode_source = RedshiftSource(
    name="redshift_zipcode_source",
    query="SELECT * FROM spectrum.zipcode_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

credit_history_source = RedshiftSource(
    name="redshift_credit_history_source",
    query="SELECT * FROM spectrum.credit_history",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)
