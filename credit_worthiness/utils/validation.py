from typing import Any, Dict, List, Mapping, Optional, Sequence


class FormValidator:
    """
    Builds the validation rules once from the parsed config/form.yaml content.
    """

    def __init__(self, fields: Sequence[Mapping[str, Any]]):
        self.fields = list(fields)
        self.field_names: List[str] = [field["name"] for field in self.fields]
        self._by_name: Dict[str, Mapping[str, Any]] = {field["name"]: field for field in self.fields}

    def missing_fields(self, payload: Mapping[str, Any]) -> List[str]:
        """
        Names of the fields the caller did not provide at all.
        """
        return [name for name in self.field_names if payload.get(name) in (None, "")]

    def _validate_select(self, field: Mapping[str, Any], value: Any) -> Optional[str]:
        options = field.get("options", [])
        if str(value) not in options:
            return (
                f"'{field['name']}' got the unsupported value {value!r}. "
                f"Allowed values: {', '.join(repr(option) for option in options)}."
            )
        return None

    def _validate_number(self, field: Mapping[str, Any], value: Any) -> Optional[str]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return f"'{field['name']}' has to be a number, got {value!r}."

        minimum, maximum = field.get("min"), field.get("max")
        if minimum is not None and number < float(minimum):
            return f"'{field['name']}' has to be greater than or equal to {minimum}, got {number:g}."
        if maximum is not None and number > float(maximum):
            return f"'{field['name']}' has to be lower than or equal to {maximum}, got {number:g}."
        return None

    def validate(self, payload: Mapping[str, Any]) -> List[str]:
        """
        Returns a human readable error per invalid field, an empty list when the
        whole payload matches config/form.yaml.
        """
        errors: List[str] = []

        missing = self.missing_fields(payload)
        if missing:
            errors.append(f"Missing values for: {', '.join(missing)}.")

        for name, value in payload.items():
            field = self._by_name.get(name)
            if field is None or value in (None, ""):
                continue

            error = (
                self._validate_number(field, value)
                if field.get("type") == "number"
                else self._validate_select(field, value)
            )
            if error:
                errors.append(error)

        return errors
