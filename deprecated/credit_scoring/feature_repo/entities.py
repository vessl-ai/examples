from feast import Entity, ValueType

zipcode = Entity(name="zipcode", value_type=ValueType.INT64)

dob_ssn = Entity(
    name="dob_ssn",
    value_type=ValueType.STRING,
    description="Date of birth and last four digits of social security number",
)
