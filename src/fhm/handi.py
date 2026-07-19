




class InputHandler:
    """
    Generic input object.

    The object can contain either a single string value
    or an iterable containing string elements.
    """

    def __init__(self, value):
        self._validate(value)

        if isinstance(value, str):
            self.values = [value]
        else:
            self.values = list(value)

    
    def _validate(self, value):
        """
        Validate input type.
        """

        if isinstance(value, str):
            return

        try:
            iterator = iter(value)
        except TypeError:
            raise TypeError("Input must be a string or an iterable of strings")

        for element in iterator:
            if not isinstance(element, str):
                raise TypeError("All iterable elements must be strings")




