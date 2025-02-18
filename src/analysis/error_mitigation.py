import itertools
import json
from typing import Tuple
import numpy as np
import pandas as pd
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit.ignis.mitigation import complete_meas_cal, CompleteMeasFitter
from typing import List, Tuple, Union
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, QiskitError
from qiskit_ibm_runtime.ibm_backend import IBMBackend

from src.analysis.constants import MATRIX, STATES
from src.observables.gauss import gauss_law, sector_2, gauss_law_squared
import re
import copy
from qiskit import QiskitError
from qiskit.result import Result
# from .filters import MeasurementFilter, TensoredFilter

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from copy import deepcopy
from scipy.optimize import minimize
import scipy.linalg as la
import numpy as np
import qiskit
# from qiskit.validation.base import Obj
from qiskit import QiskitError
# from qiskit.tools import parallel_map
from functools import wraps
from types import SimpleNamespace, MethodType
from qiskit.exceptions import QiskitError
from marshmallow import ValidationError
from marshmallow import Schema, post_dump, post_load
from marshmallow import fields as _fields
from marshmallow.utils import is_collection, INCLUDE
# from .exceptions import ModelValidationError

import os
import platform
from concurrent.futures import ProcessPoolExecutor
from qiskit.exceptions import QiskitError
# from qiskit.util import local_hardware_info

import multiprocessing as mp
import platform
import os
# from qiskit.tools.events.pubsub import Publisher

# Set parallel flag
os.environ['QISKIT_IN_PARALLEL'] = 'FALSE'

# Number of local physical cpus

#############################################################################


def local_hardware_info():
    """Basic hardware information about the local machine.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on. CPU count defaults to 1 when true count can't be determined.

    Returns:
        dict: The hardware information.
    """

    if hasattr(os, "sched_getaffinity"):
        num_cpus = len(os.sched_getaffinity(0))
    else:
        num_cpus = os.cpu_count()
    if num_cpus is None:
        num_cpus = 1
    else:
        num_cpus = int(num_cpus / 2) or 1

    results = {
        "python_compiler": platform.python_compiler(),
        "python_build": ", ".join(platform.python_build()),
        "python_version": platform.python_version(),
        "os": platform.system(),
        "cpus": num_cpus,
    }
    return results


def is_main_process():
    """Checks whether the current process is the main one"""
    if platform.system() == "Windows":
        return not isinstance(mp.current_process(), mp.context.SpawnProcess)
    else:
        return not isinstance(
            mp.current_process(), (mp.context.ForkProcess, mp.context.SpawnProcess)
        )
    
CPU_COUNT = local_hardware_info()['cpus']
class _Broker:
    """The event/message broker. It's a singleton.

    In order to keep consistency across all the components, it would be great to
    have a specific format for new events, documenting their usage.
    It's the responsibility of the component emitting an event to document it's usage in
    the component docstring.

    Event format:
        "terra.<component>.<method>.<action>"

    Examples:
        "terra.transpiler.compile.start"
        "terra.job.status.changed"
        "terra.backend.run.start"
        "terra.job.result.received"
    """

    _instance = None
    _subscribers = {}

    def __new__(cls):
        if _Broker._instance is None:
            _Broker._instance = object.__new__(cls)
        return _Broker._instance

    class _Subscription:
        def __init__(self, event, callback):
            self.event = event
            self.callback = callback

        def __eq__(self, other):
            """Overrides the default implementation"""
            if isinstance(other, self.__class__):
                return self.event == other.event and \
                       self.callback.__name__ == other.callback.__name__
            return False

    def subscribe(self, event, callback):
        """Subscribes to an event, so when it's emitted all the callbacks subscribed,
        will be executed. We are not allowing double registration.

        Args
            event (string): The event to subscribed in the form of:
                            "terra.<component>.<method>.<action>"
            callback (callable): The callback that will be executed when an event is
                                  emitted.
        """
        if not callable(callback):
            raise QiskitError("Callback is not a callable!")

        if event not in self._subscribers:
            self._subscribers[event] = []

        new_subscription = self._Subscription(event, callback)
        if new_subscription in self._subscribers[event]:
            # We are not allowing double subscription
            return False

        self._subscribers[event].append(new_subscription)
        return True

    def dispatch(self, event, *args, **kwargs):
        """Emits an event if there are any subscribers.

        Args
            event (String): The event to be emitted
            args: Arguments linked with the event
            kwargs: Named arguments linked with the event
        """
        # No event, no subscribers.
        if event not in self._subscribers:
            return

        for subscriber in self._subscribers[event]:
            subscriber.callback(*args, **kwargs)

    def unsubscribe(self, event, callback):
        """ Unsubscribe the specific callback to the event.

        Args
            event (String): The event to unsubscribe
            callback (callable): The callback that won't be executed anymore

        Returns
            True: if we have successfully unsubscribed to the event
            False: if there's no callback previously registered
        """

        try:
            self._subscribers[event].remove(self._Subscription(event, callback))
        except KeyError:
            return False

        return True

    def clear(self):
        """ Unsubscribe everything, leaving the Broker without subscribers/events.
        """
        self._subscribers.clear()


class Publisher:
    """ Represents a Publisher, every component (class) can become a Publisher and
    send events by inheriting this class. Functions can call this class like:
    Publisher().publish("event", args, ... )
    """
    def __init__(self):
        self._broker = _Broker()

    def publish(self, event, *args, **kwargs):
        """ Triggers an event, and associates some data to it, so if there are any
        subscribers, their callback will be called synchronously. """
        return self._broker.dispatch(event, *args, **kwargs)


class Subscriber:
    """ Represents a Subscriber, every component (class) can become a Subscriber and
    subscribe to events, that will call callback functions when they are emitted.
    """
    def __init__(self):
        self._broker = _Broker()

    def subscribe(self, event, callback):
        """ Subscribes to an event, associating a callback function to that event, so
        when the event occurs, the callback will be called.
        This is a blocking call, so try to keep callbacks as lightweight as possible. """
        return self._broker.subscribe(event, callback)

    def unsubscribe(self, event, callback):
        """ Unsubscribe a pair event-callback, so the callback will not be called anymore
        when the event occurs."""
        return self._broker.unsubscribe(event, callback)

    def clear(self):
        """ Unsubscribe everything"""
        self._broker.clear()

##################################################################################################
def _task_wrapper(param):
    (task, value, task_args, task_kwargs) = param
    return task(value, *task_args, **task_kwargs)

def parallel_map(  # pylint: disable=dangerous-default-value
        task, values, task_args=tuple(), task_kwargs={}, num_processes=CPU_COUNT):
    """
    Parallel execution of a mapping of `values` to the function `task`. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    On Windows this function defaults to a serial implementation to avoid the
    overhead from spawning processes in Windows.

    Args:
        task (func): Function that is to be called for each value in ``values``.
        values (array_like): List or array of values for which the ``task``
                            function is to be evaluated.
        task_args (list): Optional additional arguments to the ``task`` function.
        task_kwargs (dict): Optional additional keyword argument to the ``task`` function.
        num_processes (int): Number of processes to spawn.

    Returns:
        result: The result list contains the value of
                ``task(value, *task_args, **task_kwargs)`` for
                    each value in ``values``.

    Raises:
        QiskitError: If user interrupts via keyboard.

    Events:
        terra.parallel.start: The collection of parallel tasks are about to start.
        terra.parallel.update: One of the parallel task has finished.
        terra.parallel.finish: All the parallel tasks have finished.
    """
    if len(values) == 1:
        return [task(values[0], *task_args, **task_kwargs)]

    Publisher().publish("terra.parallel.start", len(values))
    nfinished = [0]

    def _callback(_):
        nfinished[0] += 1
        Publisher().publish("terra.parallel.done", nfinished[0])

    # Run in parallel if not Win and not in parallel already
    if platform.system() != 'Windows' and num_processes > 1 \
       and os.getenv('QISKIT_IN_PARALLEL') == 'FALSE':
        os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'
        try:
            results = []
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                param = map(lambda value: (task, value, task_args, task_kwargs), values)
                future = executor.map(_task_wrapper, param)

            results = list(future)
            Publisher().publish("terra.parallel.done", len(results))

        except (KeyboardInterrupt, Exception) as error:
            if isinstance(error, KeyboardInterrupt):
                Publisher().publish("terra.parallel.finish")
                os.environ['QISKIT_IN_PARALLEL'] = 'FALSE'
                raise QiskitError('Keyboard interrupt in parallel_map.')
            # Otherwise just reset parallel flag and error
            os.environ['QISKIT_IN_PARALLEL'] = 'FALSE'
            raise error

        Publisher().publish("terra.parallel.finish")
        os.environ['QISKIT_IN_PARALLEL'] = 'FALSE'
        return results

    # Cannot do parallel on Windows , if another parallel_map is running in parallel,
    # or len(values) == 1.
    results = []
    for _, value in enumerate(values):
        result = task(value, *task_args, **task_kwargs)
        results.append(result)
        _callback(0)
    Publisher().publish("terra.parallel.finish")
    return results

##########################################################################################################
class ModelValidationError(QiskitError, ValidationError):
    """Raised when a sequence subscript is out of range."""
    def __init__(self, message, field_name=None, data=None, valid_data=None,
                 **kwargs):
        # pylint: disable=super-init-not-called
        # ValidationError.__init__ is called manually instead of calling super,
        # as the signatures of ValidationError and QiskitError constructors
        # differ.
        ValidationError.__init__(self, message, field_name, data, valid_data, **kwargs)
        self.message = str(message)

#############################################################################
class ModelTypeValidator(_fields.Field):
    """A field able to validate the correct type of a value."""

    valid_types = (object, )

    def _expected_types(self):
        return self.valid_types

    def check_type(self, value, attr, data, **_):
        """Validates a value against the correct type of the field.

        It calls ``_expected_types`` to get a list of valid types.

        Subclasses can do one of the following:

        1. Override the ``valid_types`` property with a tuple with the expected
           types for this field.

        2. Override the ``_expected_types`` method to return a tuple of
           expected types for the field.

        3. Change ``check_type`` completely to customize validation.

        Note:
            This method or the overrides must return the ``value`` parameter
            untouched.
        """
        expected_types = self._expected_types()
        if not isinstance(value, expected_types):
            raise self._not_expected_type(
                value, expected_types, fields=[self], field_names=attr, data=data)
        return value

    @staticmethod
    def _not_expected_type(value, type_, **kwargs):
        if is_collection(type_) and len(type_) == 1:
            type_ = type_[0]

        if is_collection(type_):
            body = 'is none of the expected types {}'.format(type_)
        else:
            body = 'is not the expected type {}'.format(type_)

        message = 'Value \'{}\' {}: {}'.format(value, type(value), body)
        return ValidationError(message, **kwargs)

    def make_error_serialize(self, key, **kwargs):
        """Helper method to return a ValidationError from _serialize.

        This method wraps the result of ``make_error()``, adding contextual
        information in order to provide more informative information to users.

        Args:
            key (str): error key index.
            **kwargs: additional arguments to ``make_error()``.

        Returns:
            ValidationError: an exception with the field name.
        """
        bare_error = self.make_error(key, **kwargs)
        return ValidationError({self.name: bare_error.messages},
                               field_name=self.name)


class BaseSchema(Schema):
    """Base class for Schemas for validated Qiskit classes.

    Provides convenience functionality for the Qiskit common use case:

    * deserialization into class instances instead of dicts.
    * handling of unknown attributes not defined in the schema.

    Attributes:
         model_cls (type): class used to instantiate the instance. The
         constructor is passed all named parameters from deserialization.
    """

    class Meta:
        """Add extra fields to the schema."""
        unknown = INCLUDE

    model_cls = SimpleNamespace

    @post_dump(pass_original=True, pass_many=True)
    def dump_additional_data(self, valid_data, original_data, **kwargs):
        """Include unknown fields after dumping.

        Unknown fields are added with no processing at all.

        Args:
            valid_data (dict or list): data collected and returned by ``dump()``.
            original_data (object or list): object passed to ``dump()`` in the
                first place.
            **kwargs: extra arguments from the decorators.

        Returns:
            dict: the same ``valid_data`` extended with the unknown attributes.

        Inspired by https://github.com/marshmallow-code/marshmallow/pull/595.
        """
        if kwargs.get('many'):
            for i, _ in enumerate(valid_data):
                additional_keys = set(original_data[i].__dict__) - set(valid_data[i])
                for key in additional_keys:
                    if key.startswith('_'):
                        continue
                    valid_data[i][key] = getattr(original_data[i], key)
        else:
            additional_keys = set(original_data.__dict__) - set(valid_data)
            for key in additional_keys:
                if key.startswith('_'):
                    continue
                valid_data[key] = getattr(original_data, key)

        return valid_data

    @post_load
    def make_model(self, data, **_):
        """Make ``load`` return a ``model_cls`` instance instead of a dict."""
        return self.model_cls(**data)


class _SchemaBinder:
    """Helper class for the parametrized decorator ``bind_schema``."""

    def __init__(self, schema_cls, **kwargs):
        """Get the schema for the decorated model."""
        self._schema_cls = schema_cls
        self._kwargs = kwargs

    def __call__(self, model_cls):
        """Augment the model class with the validation API.

        See the docs for ``bind_schema`` for further information.
        """
        # Check for double binding of schemas.
        if self._schema_cls.__dict__.get('model_cls', None) is not None:
            raise ValueError(
                'The schema {} can not be bound twice. It is already bound to '
                '{}. If you want to reuse the schema, use '
                'subclassing'.format(self._schema_cls, self._schema_cls.model_cls))

        # Set a reference to the Model in the Schema, and vice versa.
        self._schema_cls.model_cls = model_cls
        model_cls.schema = self._schema_cls(**self._kwargs)

        # Append the methods to the Model class.
        model_cls.__init__ = self._validate_after_init(model_cls.__init__)

        # Add a Schema that performs minimal validation to the Model.
        model_cls.shallow_schema = self._create_validation_schema(self._schema_cls)

        return model_cls

    @staticmethod
    def _create_validation_schema(schema_cls, **kwargs):
        """Create a patched Schema for validating models.

        Model validation is not part of Marshmallow. Schemas have a ``validate``
        method but this delegates execution on ``load``. Similarly, ``load``
        will call ``_deserialize`` on every field in the schema.

        This function patches the ``_deserialize`` instance method of each
        field to make it call a custom defined method ``check_type``
        provided by Qiskit in the different fields at
        ``qiskit.validation.fields``.

        Returns:
            BaseSchema: a copy of the original Schema, overriding the
                ``_deserialize()`` call of its fields.
        """
        validation_schema = schema_cls(**kwargs)
        for _, field in validation_schema.fields.items():
            if isinstance(field, ModelTypeValidator):
                validate_function = field.__class__.check_type
                field._deserialize = MethodType(validate_function, field)

        return validation_schema

    @staticmethod
    def _validate_after_init(init_method):
        """Add validation during instantiation.

        The validation is performed depending on the ``validate`` parameter
        passed to the ``init_method``. If ``False``, the validation will not be
        performed.
        """
        @wraps(init_method)
        def _decorated(self, **kwargs):
            # Extract the 'validate' parameter.
            do_validation = kwargs.pop('validate', True)
            if do_validation:
                try:
                    _ = self.shallow_schema._do_load(kwargs, postprocess=False)
                except ValidationError as ex:
                    raise ModelValidationError(
                        ex.messages, ex.field_name, ex.data, ex.valid_data, **ex.kwargs) from None

            # Set the 'validate' parameter to False, assuming that if a
            # subclass has been validated, it superclasses will also be valid.
            return init_method(self, **kwargs, validate=False)

        return _decorated


def bind_schema(schema, **kwargs):
    """Class decorator for adding schema validation to its instances.

    The decorator acts on the model class by adding:

    * a class attribute ``schema`` with the schema used for validation
    * a class attribute ``shallow_schema`` used for validation during
      instantiation.

    The same schema cannot be bound more than once. If you need to reuse a
    schema for a different class, create a new schema subclassing the one you
    want to reuse and leave the new empty::

        class MySchema(BaseSchema):
            title = String()

        class AnotherSchema(MySchema):
            pass

        @bind_schema(MySchema):
        class MyModel(BaseModel):
            pass

        @bind_schema(AnotherSchema):
        class AnotherModel(BaseModel):
            pass

    Note:
        By default, models decorated with this decorator are validated during
        instantiation. If ``validate=False`` is passed to the constructor, this
        validation will not be performed.

    Args:
        schema (class): the schema class used for validation.
        **kwargs: Additional attributes for the ``marshmallow.Schema``
            initializer.

    Raises:
        ValueError: when trying to bind the same schema more than once.

    Return:
        type: the same class with validation capabilities.
    """
    return _SchemaBinder(schema, **kwargs)


def _base_model_from_kwargs(cls, kwargs):
    """Helper for BaseModel.__reduce__, expanding kwargs."""
    return cls(**kwargs)


class BaseModel(SimpleNamespace):
    """Base class for Models for validated Qiskit classes."""

    def __init__(self, validate=True, **kwargs):
        """BaseModel initializer.

        Note:
            The ``validate`` argument is used for controlling the behavior of
            the schema binding, and will not be present on the created object.
        """
        # pylint: disable=unused-argument
        super().__init__(**kwargs)

    def __reduce__(self):
        """Custom __reduce__ for allowing pickling and unpickling.

        Customize the reduction in order to allow serialization, as the
        BaseModels need to be pickled during the use of futures by the backends.
        Instead of returning the class, a helper is used in order to pass the
        arguments as **kwargs, as it is needed by SimpleNamespace and the
        standard __reduce__ only allows passing args as a tuple.
        """
        return _base_model_from_kwargs, (self.__class__, self.__dict__)

    def __contains__(self, item):
        """Custom implementation of membership test.

        Implement the ``__contains__`` method for catering to the common case
        of finding out if a model contains a certain key (``key in model``).
        """
        return item in self.__dict__

    def to_dict(self):
        """Serialize the model into a Python dict of simple types.

        Note that this method requires that the model is bound with
        ``@bind_schema``.
        """
        try:
            data = self.schema.dump(self)
        except ValidationError as ex:
            raise ModelValidationError(
                ex.messages, ex.field_name, ex.data, ex.valid_data, **ex.kwargs) from None

        return data

    @classmethod
    def from_dict(cls, dict_):
        """Deserialize a dict of simple types into an instance of this class.

        Note that this method requires that the model is bound with
        ``@bind_schema``.
        """
        try:
            data = cls.schema.load(dict_)
        except ValidationError as ex:
            raise ModelValidationError(
                ex.messages, ex.field_name, ex.data, ex.valid_data, **ex.kwargs) from None

        return data


class ObjSchema(BaseSchema):
    """Generic object schema."""
    pass


@bind_schema(ObjSchema)
class Obj(BaseModel):
    """Generic object in a Model."""
    pass

###############################################################
def count_keys(num_qubits: int) -> List[str]:
    """Return ordered count keys.

    Args:
        num_qubits: The number of qubits in the generated list.
    Returns:
        The strings of all 0/1 combinations of the given number of qubits
    Example:
        >>> count_keys(3)
        ['000', '001', '010', '011', '100', '101', '110', '111']
    """
    return [bin(j)[2:].zfill(num_qubits)
            for j in range(2 ** num_qubits)]

def complete_meas_cal(qubit_list: List[int] = None,
                      qr: Union[int, List[QuantumRegister]] = None,
                      cr: Union[int, List[ClassicalRegister]] = None,
                      circlabel: str = ''
                      ) -> Tuple[List[QuantumCircuit], List[str]
                                 ]:
    """
    Return a list of measurement calibration circuits for the full
    Hilbert space.

    If the circuit contains :math:`n` qubits, then :math:`2^n` calibration circuits
    are created, each of which creates a basis state.

    Args:
        qubit_list: A list of qubits to perform the measurement correction on.
           If `None`, and qr is given then assumed to be performed over the entire
           qr. The calibration states will be labelled according to this ordering (default `None`).

        qr: Quantum registers (or their size).
        If `None`, one is created (default `None`).

        cr: Classical registers (or their size).
        If `None`, one is created(default `None`).

        circlabel: A string to add to the front of circuit names for
            unique identification(default ' ').

    Returns:
        A list of QuantumCircuit objects containing the calibration circuits.

        A list of calibration state labels.

    Additional Information:
        The returned circuits are named circlabel+cal_XXX
        where XXX is the basis state,
        e.g., cal_1001.

        Pass the results of these circuits to the CompleteMeasurementFitter
        constructor.

    Raises:
        QiskitError: if both `qubit_list` and `qr` are `None`.

    """

    if qubit_list is None and qr is None:
        raise QiskitError("Must give one of a qubit_list or a qr")

    # Create the registers if not already done
    if qr is None:
        qr = QuantumRegister(max(qubit_list)+1)

    if isinstance(qr, int):
        qr = QuantumRegister(qr)

    if qubit_list is None:
        qubit_list = range(len(qr))

    if isinstance(cr, int):
        cr = ClassicalRegister(cr)

    nqubits = len(qubit_list)

    # labels for 2**n qubit states
    state_labels = count_keys(nqubits)

    cal_circuits, _ = tensored_meas_cal([qubit_list],
                                        qr, cr, circlabel)

    return cal_circuits, state_labels


def tensored_meas_cal(mit_pattern: List[List[int]] = None,
                      qr: Union[int, List[QuantumRegister]] = None,
                      cr: Union[int, List[ClassicalRegister]] = None,
                      circlabel: str = ''
                      ) -> Tuple[List[QuantumCircuit], List[List[int]]
                                 ]:
    """
    Return a list of calibration circuits

    Args:
        mit_pattern: Qubits on which to perform the
            measurement correction, divided to groups according to tensors.
            If `None` and `qr` is given then assumed to be performed over the entire
            `qr` as one group (default `None`).

        qr: A quantum register (or its size).
        If `None`, one is created (default `None`).

        cr: A classical register (or its size).
        If `None`, one is created (default `None`).

        circlabel: A string to add to the front of circuit names for
            unique identification (default ' ').

    Returns:
        A list of two QuantumCircuit objects containing the calibration circuits
        mit_pattern

    Additional Information:
        The returned circuits are named circlabel+cal_XXX
        where XXX is the basis state,
        e.g., cal_000 and cal_111.

        Pass the results of these circuits to the TensoredMeasurementFitter
        constructor.

    Raises:
        QiskitError: if both `mit_pattern` and `qr` are None.
        QiskitError: if a qubit appears more than once in `mit_pattern`.

    """

    if mit_pattern is None and qr is None:
        raise QiskitError("Must give one of mit_pattern or qr")

    if isinstance(qr, int):
        qr = QuantumRegister(qr)

    qubits_in_pattern = []
    if mit_pattern is not None:
        for qubit_list in mit_pattern:
            for qubit in qubit_list:
                if qubit in qubits_in_pattern:
                    raise QiskitError("mit_pattern cannot contain \
                    multiple instances of the same qubit")
                qubits_in_pattern.append(qubit)

        # Create the registers if not already done
        if qr is None:
            qr = QuantumRegister(max(qubits_in_pattern)+1)
    else:
        qubits_in_pattern = range(len(qr))
        mit_pattern = [qubits_in_pattern]

    nqubits = len(qubits_in_pattern)

    # create classical bit registers
    if cr is None:
        cr = ClassicalRegister(nqubits)

    if isinstance(cr, int):
        cr = ClassicalRegister(cr)

    qubits_list_sizes = [len(qubit_list) for qubit_list in mit_pattern]
    nqubits = sum(qubits_list_sizes)
    size_of_largest_group = max(qubits_list_sizes)
    largest_labels = count_keys(size_of_largest_group)

    state_labels = []
    for largest_state in largest_labels:
        basis_state = ''
        for list_size in qubits_list_sizes:
            basis_state = largest_state[:list_size] + basis_state
        state_labels.append(basis_state)

    cal_circuits = []
    for basis_state in state_labels:
        qc_circuit = QuantumCircuit(qr, cr,
                                    name='%scal_%s' % (circlabel, basis_state))

        end_index = nqubits
        for qubit_list, list_size in zip(mit_pattern, qubits_list_sizes):

            start_index = end_index - list_size
            substate = basis_state[start_index:end_index]

            for qind in range(list_size):
                if substate[list_size-qind-1] == '1':
                    qc_circuit.x(qr[qubit_list[qind]])

            end_index = start_index

        qc_circuit.barrier(qr)

        # add measurements
        end_index = nqubits
        for qubit_list, list_size in zip(mit_pattern, qubits_list_sizes):

            for qind in range(list_size):
                qc_circuit.measure(qr[qubit_list[qind]],
                                   cr[nqubits-(end_index-qind)])

            end_index -= list_size

        cal_circuits.append(qc_circuit)

    return cal_circuits, mit_pattern

###########################################################################################
class MeasurementFilter():
    """
    Measurement error mitigation filter.

    Produced from a measurement calibration fitter and can be applied
    to data.

    """

    def __init__(self,
                 cal_matrix: np.matrix,
                 state_labels: list):
        """
        Initialize a measurement error mitigation filter using the cal_matrix
        from a measurement calibration fitter.

        Args:
            cal_matrix: the calibration matrix for applying the correction
            state_labels: the states for the ordering of the cal matrix
        """

        self._cal_matrix = cal_matrix
        self._state_labels = state_labels

    @property
    def cal_matrix(self):
        """Return cal_matrix."""
        return self._cal_matrix

    @property
    def state_labels(self):
        """return the state label ordering of the cal matrix"""
        return self._state_labels

    @state_labels.setter
    def state_labels(self, new_state_labels):
        """set the state label ordering of the cal matrix"""
        self._state_labels = new_state_labels

    @cal_matrix.setter
    def cal_matrix(self, new_cal_matrix):
        """Set cal_matrix."""
        self._cal_matrix = new_cal_matrix

    def apply(self,
              raw_data,
              method='least_squares'):
        """Apply the calibration matrix to results.

        Args:
            raw_data (dict or list): The data to be corrected. Can be in a number of forms:

                 Form 1: a counts dictionary from results.get_counts

                 Form 2: a list of counts of `length==len(state_labels)`

                 Form 3: a list of counts of `length==M*len(state_labels)` where M is an
                 integer (e.g. for use with the tomography data)

                 Form 4: a qiskit Result

            method (str): fitting method. If `None`, then least_squares is used.

                ``pseudo_inverse``: direct inversion of the A matrix

                ``least_squares``: constrained to have physical probabilities

        Returns:
            dict or list: The corrected data in the same form as `raw_data`

        Raises:
            QiskitError: if `raw_data` is not an integer multiple
                of the number of calibrated states.

        """

        # check forms of raw_data
        if isinstance(raw_data, dict):
            # counts dictionary
            for data_label in raw_data.keys():
                if data_label not in self._state_labels:
                    raise QiskitError("Unexpected state label '" + data_label +
                                      "', verify the fitter's state labels "
                                      "correpsond to the input data")
            data_format = 0
            # convert to form2
            raw_data2 = [np.zeros(len(self._state_labels), dtype=float)]
            for stateidx, state in enumerate(self._state_labels):
                raw_data2[0][stateidx] = raw_data.get(state, 0)

        elif isinstance(raw_data, list):
            size_ratio = len(raw_data)/len(self._state_labels)
            if len(raw_data) == len(self._state_labels):
                data_format = 1
                raw_data2 = [raw_data]
            elif int(size_ratio) == size_ratio:
                data_format = 2
                size_ratio = int(size_ratio)
                # make the list into chunks the size of state_labels for easier
                # processing
                raw_data2 = np.zeros([size_ratio, len(self._state_labels)])
                for i in range(size_ratio):
                    raw_data2[i][:] = raw_data[
                        i * len(self._state_labels):(i + 1)*len(
                            self._state_labels)]
            else:
                raise QiskitError("Data list is not an integer multiple "
                                  "of the number of calibrated states")

        elif isinstance(raw_data, qiskit.result.result.Result):

            # extract out all the counts, re-call the function with the
            # counts and push back into the new result
            new_result = deepcopy(raw_data)

            new_counts_list = parallel_map(
                self._apply_correction,
                [resultidx for resultidx, _ in enumerate(raw_data.results)],
                task_args=(raw_data, method))

            for resultidx, new_counts in new_counts_list:
                new_result.results[resultidx].data.counts = Obj(**new_counts)

            return new_result

        else:
            raise QiskitError("Unrecognized type for raw_data.")

        if method == 'pseudo_inverse':
            pinv_cal_mat = la.pinv(self._cal_matrix)

        # Apply the correction
        for data_idx, _ in enumerate(raw_data2):

            if method == 'pseudo_inverse':
                raw_data2[data_idx] = np.dot(
                    pinv_cal_mat, raw_data2[data_idx])

            elif method == 'least_squares':
                nshots = sum(raw_data2[data_idx])

                def fun(x):
                    return sum(
                        (raw_data2[data_idx] - np.dot(self._cal_matrix, x))**2)
                x0 = np.random.rand(len(self._state_labels))
                x0 = x0 / sum(x0)
                cons = ({'type': 'eq', 'fun': lambda x: nshots - sum(x)})
                bnds = tuple((0, nshots) for x in x0)
                res = minimize(fun, x0, method='SLSQP',
                               constraints=cons, bounds=bnds, tol=1e-6)
                raw_data2[data_idx] = res.x

            else:
                raise QiskitError("Unrecognized method.")

        if data_format == 2:
            # flatten back out the list
            raw_data2 = raw_data2.flatten()

        elif data_format == 0:
            # convert back into a counts dictionary
            new_count_dict = {}
            for stateidx, state in enumerate(self._state_labels):
                if raw_data2[0][stateidx] != 0:
                    new_count_dict[state] = raw_data2[0][stateidx]

            raw_data2 = new_count_dict
        else:
            # TODO: should probably change to:
            # raw_data2 = raw_data2[0].tolist()
            raw_data2 = raw_data2[0]
        return raw_data2

    def _apply_correction(self, resultidx, raw_data, method):
        """Wrapper to call apply with a counts dictionary."""
        new_counts = self.apply(
            raw_data.get_counts(resultidx), method=method)
        return resultidx, new_counts


class TensoredFilter():
    """
    Tensored measurement error mitigation filter.

    Produced from a tensored measurement calibration fitter and can be applied
    to data.
    """

    def __init__(self,
                 cal_matrices: np.matrix,
                 substate_labels_list: list):
        """
        Initialize a tensored measurement error mitigation filter using
        the cal_matrices from a tensored measurement calibration fitter.

        Args:
            cal_matrices: the calibration matrices for applying the correction.
            substate_labels_list: for each calibration matrix
                a list of the states (as strings, states in the subspace)
        """

        self._cal_matrices = cal_matrices
        self._qubit_list_sizes = []
        self._indices_list = []
        self._substate_labels_list = []
        self.substate_labels_list = substate_labels_list

    @property
    def cal_matrices(self):
        """Return cal_matrices."""
        return self._cal_matrices

    @cal_matrices.setter
    def cal_matrices(self, new_cal_matrices):
        """Set cal_matrices."""
        self._cal_matrices = deepcopy(new_cal_matrices)

    @property
    def substate_labels_list(self):
        """Return _substate_labels_list"""
        return self._substate_labels_list

    @substate_labels_list.setter
    def substate_labels_list(self, new_substate_labels_list):
        """Return _substate_labels_list"""
        self._substate_labels_list = new_substate_labels_list

        # get the number of qubits in each subspace
        self._qubit_list_sizes = []
        for _, substate_label_list in enumerate(self._substate_labels_list):
            self._qubit_list_sizes.append(
                int(np.log2(len(substate_label_list))))

        # get the indices in the calibration matrix
        self._indices_list = []
        for _, sub_labels in enumerate(self._substate_labels_list):

            self._indices_list.append(
                {lab: ind for ind, lab in enumerate(sub_labels)})

    @property
    def qubit_list_sizes(self):
        """Return _qubit_list_sizes."""
        return self._qubit_list_sizes

    @property
    def nqubits(self):
        """Return the number of qubits. See also MeasurementFilter.apply() """
        return sum(self._qubit_list_sizes)

    def apply(self, raw_data, method='least_squares'):
        """
        Apply the calibration matrices to results.

        Args:
            raw_data (dict or Result): The data to be corrected. Can be in one of two forms:

                * A counts dictionary from results.get_counts

                * A Qiskit Result

            method (str): fitting method. The following methods are supported:

                * 'pseudo_inverse': direct inversion of the cal matrices.

                * 'least_squares': constrained to have physical probabilities.

                * If `None`, 'least_squares' is used.

        Returns:
            dict or Result: The corrected data in the same form as raw_data

        Raises:
            QiskitError: if raw_data is not in a one of the defined forms.
        """

        all_states = count_keys(self.nqubits)
        num_of_states = 2**self.nqubits

        # check forms of raw_data
        if isinstance(raw_data, dict):
            # counts dictionary
            # convert to list
            raw_data2 = [np.zeros(num_of_states, dtype=float)]
            for state, count in raw_data.items():
                stateidx = int(state, 2)
                raw_data2[0][stateidx] = count

        elif isinstance(raw_data, qiskit.result.result.Result):

            # extract out all the counts, re-call the function with the
            # counts and push back into the new result
            new_result = deepcopy(raw_data)

            new_counts_list = parallel_map(
                self._apply_correction,
                [resultidx for resultidx, _ in enumerate(raw_data.results)],
                task_args=(raw_data, method))

            for resultidx, new_counts in new_counts_list:
                new_result.results[resultidx].data.counts = Obj(**new_counts)

            return new_result

        else:
            raise QiskitError("Unrecognized type for raw_data.")

        if method == 'pseudo_inverse':
            pinv_cal_matrices = []
            for cal_mat in self._cal_matrices:
                pinv_cal_matrices.append(la.pinv(cal_mat))

        # Apply the correction
        for data_idx, _ in enumerate(raw_data2):

            if method == 'pseudo_inverse':
                inv_mat_dot_raw = np.zeros([num_of_states], dtype=float)
                for state1_idx, state1 in enumerate(all_states):
                    for state2_idx, state2 in enumerate(all_states):
                        if raw_data2[data_idx][state2_idx] == 0:
                            continue

                        product = 1.
                        end_index = self.nqubits
                        for p_ind, pinv_mat in enumerate(pinv_cal_matrices):

                            start_index = end_index - \
                                self._qubit_list_sizes[p_ind]

                            state1_as_int = \
                                self._indices_list[p_ind][
                                    state1[start_index:end_index]]

                            state2_as_int = \
                                self._indices_list[p_ind][
                                    state2[start_index:end_index]]

                            end_index = start_index
                            product *= \
                                pinv_mat[state1_as_int][state2_as_int]
                            if product == 0:
                                break
                        inv_mat_dot_raw[state1_idx] += \
                            (product * raw_data2[data_idx][state2_idx])
                raw_data2[data_idx] = inv_mat_dot_raw

            elif method == 'least_squares':

                def fun(x):
                    mat_dot_x = np.zeros([num_of_states], dtype=float)
                    for state1_idx, state1 in enumerate(all_states):
                        mat_dot_x[state1_idx] = 0.
                        for state2_idx, state2 in enumerate(all_states):
                            if x[state2_idx] != 0:
                                product = 1.
                                end_index = self.nqubits
                                for c_ind, cal_mat in \
                                        enumerate(self._cal_matrices):

                                    start_index = end_index - \
                                        self._qubit_list_sizes[c_ind]

                                    state1_as_int = \
                                        self._indices_list[c_ind][
                                            state1[start_index:end_index]]

                                    state2_as_int = \
                                        self._indices_list[c_ind][
                                            state2[start_index:end_index]]

                                    end_index = start_index
                                    product *= \
                                        cal_mat[state1_as_int][state2_as_int]
                                    if product == 0:
                                        break
                                mat_dot_x[state1_idx] += \
                                    (product * x[state2_idx])
                    return sum(
                        (raw_data2[data_idx] - mat_dot_x)**2)

                x0 = np.random.rand(num_of_states)
                x0 = x0 / sum(x0)
                nshots = sum(raw_data2[data_idx])
                cons = ({'type': 'eq', 'fun': lambda x: nshots - sum(x)})
                bnds = tuple((0, nshots) for x in x0)
                res = minimize(fun, x0, method='SLSQP',
                               constraints=cons, bounds=bnds, tol=1e-6)
                raw_data2[data_idx] = res.x

            else:
                raise QiskitError("Unrecognized method.")

        # convert back into a counts dictionary
        new_count_dict = {}
        for state_idx, state in enumerate(all_states):
            if raw_data2[0][state_idx] != 0:
                new_count_dict[state] = raw_data2[0][state_idx]

        return new_count_dict

    def _apply_correction(self, resultidx, raw_data, method):
        """Wrapper to call apply with a counts dictionary."""
        new_counts = self.apply(
            raw_data.get_counts(resultidx), method=method)
        return resultidx, new_counts
    
#################################################################################################

class CompleteMeasFitter:
    """
    Measurement correction fitter for a full calibration
    """

    def __init__(self,
                 results: Union[Result, List[Result]],
                 state_labels: List[str],
                 qubit_list: List[int] = None,
                 circlabel: str = ''):
        """
        Initialize a measurement calibration matrix from the results of running
        the circuits returned by `measurement_calibration_circuits`

        A wrapper for the tensored fitter

        Args:
            results: the results of running the measurement calibration
                circuits. If this is `None` the user will set a calibration
                matrix later.
            state_labels: list of calibration state labels
                returned from `measurement_calibration_circuits`.
                The output matrix will obey this ordering.
            qubit_list: List of the qubits (for reference and if the
                subset is needed). If `None`, the qubit_list will be
                created according to the length of state_labels[0].
            circlabel: if the qubits were labeled.
        """
        if qubit_list is None:
            qubit_list = range(len(state_labels[0]))
        self._qubit_list = qubit_list

        self._tens_fitt = TensoredMeasFitter(results,
                                             [qubit_list],
                                             [state_labels],
                                             circlabel)

    @property
    def cal_matrix(self):
        """Return cal_matrix."""
        return self._tens_fitt.cal_matrices[0]

    @cal_matrix.setter
    def cal_matrix(self, new_cal_matrix):
        """set cal_matrix."""
        self._tens_fitt.cal_matrices = [copy.deepcopy(new_cal_matrix)]

    @property
    def state_labels(self):
        """Return state_labels."""
        return self._tens_fitt.substate_labels_list[0]

    @property
    def qubit_list(self):
        """Return list of qubits."""
        return self._qubit_list

    @state_labels.setter
    def state_labels(self, new_state_labels):
        """Set state label."""
        self._tens_fitt.substate_labels_list[0] = new_state_labels

    @property
    def filter(self):
        """Return a measurement filter using the cal matrix."""
        return MeasurementFilter(self.cal_matrix, self.state_labels)

    def add_data(self, new_results, rebuild_cal_matrix=True):
        """
        Add measurement calibration data

        Args:
            new_results (list or qiskit.result.Result): a single result or list
                of result objects.
            rebuild_cal_matrix (bool): rebuild the calibration matrix
        """

        self._tens_fitt.add_data(new_results, rebuild_cal_matrix)

    def subset_fitter(self, qubit_sublist=None):
        """
        Return a fitter object that is a subset of the qubits in the original
        list.

        Args:
            qubit_sublist (list): must be a subset of qubit_list

        Returns:
            CompleteMeasFitter: A new fitter that has the calibration for a
                subset of qubits

        Raises:
            QiskitError: If the calibration matrix is not initialized
        """

        if self._tens_fitt.cal_matrices is None:
            raise QiskitError("Calibration matrix is not initialized")

        if qubit_sublist is None:
            raise QiskitError("Qubit sublist must be specified")

        for qubit in qubit_sublist:
            if qubit not in self._qubit_list:
                raise QiskitError("Qubit not in the original set of qubits")

        # build state labels
        new_state_labels = count_keys(len(qubit_sublist))

        # mapping between indices in the state_labels and the qubits in
        # the sublist
        qubit_sublist_ind = []
        for sqb in qubit_sublist:
            for qbind, qubit in enumerate(self._qubit_list):
                if qubit == sqb:
                    qubit_sublist_ind.append(qbind)

        # states in the full calibration which correspond
        # to the reduced labels
        q_q_mapping = []
        state_labels_reduced = []
        for label in self.state_labels:
            tmplabel = [label[index] for index in qubit_sublist_ind]
            state_labels_reduced.append(''.join(tmplabel))

        for sub_lab_ind, _ in enumerate(new_state_labels):
            q_q_mapping.append([])
            for labelind, label in enumerate(state_labels_reduced):
                if label == new_state_labels[sub_lab_ind]:
                    q_q_mapping[-1].append(labelind)

        new_fitter = CompleteMeasFitter(results=None,
                                        state_labels=new_state_labels,
                                        qubit_list=qubit_sublist)

        new_cal_matrix = np.zeros([len(new_state_labels),
                                   len(new_state_labels)])

        # do a partial trace
        for i in range(len(new_state_labels)):
            for j in range(len(new_state_labels)):

                for q_q_i_map in q_q_mapping[i]:
                    for q_q_j_map in q_q_mapping[j]:
                        new_cal_matrix[i, j] += self.cal_matrix[q_q_i_map,
                                                                q_q_j_map]

                new_cal_matrix[i, j] /= len(q_q_mapping[i])

        new_fitter.cal_matrix = new_cal_matrix

        return new_fitter

    def readout_fidelity(self, label_list=None):
        """
        Based on the results, output the readout fidelity which is the
        normalized trace of the calibration matrix

        Args:
            label_list (bool): If `None`, returns the average assignment fidelity
                of a single state. Otherwise it returns the assignment fidelity
                to be in any one of these states averaged over the second
                index.

        Returns:
            numpy.array: readout fidelity (assignment fidelity)

        Additional Information:
            The on-diagonal elements of the calibration matrix are the
            probabilities of measuring state 'x' given preparation of state
            'x' and so the normalized trace is the average assignment fidelity
        """
        return self._tens_fitt.readout_fidelity(0, label_list)

    def plot_calibration(self, ax=None, show_plot=True):
        """
        Plot the calibration matrix (2D color grid plot)

        Args:
            show_plot (bool): call plt.show()
            ax (matplotlib.axes.Axes): An optional Axes object to use for the
                plot
        """

        self._tens_fitt.plot_calibration(0, ax, show_plot)


class TensoredMeasFitter():
    """
    Measurement correction fitter for a tensored calibration.
    """

    def __init__(self,
                 results: Union[Result, List[Result]],
                 mit_pattern: List[List[int]],
                 substate_labels_list: List[List[str]] = None,
                 circlabel: str = ''):
        """
        Initialize a measurement calibration matrix from the results of running
        the circuits returned by `measurement_calibration_circuits`.

        Args:
            results: the results of running the measurement calibration
                circuits. If this is `None`, the user will set calibration
                matrices later.

            mit_pattern: qubits to perform the
                measurement correction on, divided to groups according to
                tensors

            substate_labels_list: for each
                calibration matrix, the labels of its rows and columns.
                If `None`, the labels are ordered lexicographically

            circlabel: if the qubits were labeled

        Raises:
            ValueError: if the mit_pattern doesn't match the
                substate_labels_list
        """

        self._result_list = []
        self._cal_matrices = None
        self._circlabel = circlabel

        self._qubit_list_sizes = \
            [len(qubit_list) for qubit_list in mit_pattern]

        self._indices_list = []
        if substate_labels_list is None:
            self._substate_labels_list = []
            for list_size in self._qubit_list_sizes:
                self._substate_labels_list.append(count_keys(list_size))
        else:
            self._substate_labels_list = substate_labels_list
            if len(self._qubit_list_sizes) != len(substate_labels_list):
                raise ValueError("mit_pattern does not match \
                    substate_labels_list")

        self._indices_list = []
        for _, sub_labels in enumerate(self._substate_labels_list):
            self._indices_list.append(
                {lab: ind for ind, lab in enumerate(sub_labels)})

        self.add_data(results)

    @property
    def cal_matrices(self):
        """Return cal_matrices."""
        return self._cal_matrices

    @cal_matrices.setter
    def cal_matrices(self, new_cal_matrices):
        """Set _cal_matrices."""
        self._cal_matrices = copy.deepcopy(new_cal_matrices)

    @property
    def substate_labels_list(self):
        """Return _substate_labels_list."""
        return self._substate_labels_list

    @property
    def filter(self):
        """Return a measurement filter using the cal matrices."""
        return TensoredFilter(self._cal_matrices, self._substate_labels_list)

    @property
    def nqubits(self):
        """Return _qubit_list_sizes."""
        return sum(self._qubit_list_sizes)

    def add_data(self, new_results, rebuild_cal_matrix=True):
        """
        Add measurement calibration data

        Args:
            new_results (list or qiskit.result.Result): a single result or list
                of Result objects.
            rebuild_cal_matrix (bool): rebuild the calibration matrix
        """

        if new_results is None:
            return

        if not isinstance(new_results, list):
            new_results = [new_results]

        for result in new_results:
            self._result_list.append(result)

        if rebuild_cal_matrix:
            self._build_calibration_matrices()

    def readout_fidelity(self, cal_index=0, label_list=None):
        """
        Based on the results, output the readout fidelity, which is the average
        of the diagonal entries in the calibration matrices.

        Args:
            cal_index(integer): readout fidelity for this index in _cal_matrices
            label_list (list):  Returns the average fidelity over of the groups
                f states. In the form of a list of lists of states. If `None`,
                then each state used in the construction of the calibration
                matrices forms a group of size 1

        Returns:
            numpy.array: The readout fidelity (assignment fidelity)

        Raises:
            QiskitError: If the calibration matrix has not been set for the
                object.

        Additional Information:
            The on-diagonal elements of the calibration matrices are the
            probabilities of measuring state 'x' given preparation of state
            'x'.
        """

        if self._cal_matrices is None:
            raise QiskitError("Cal matrix has not been set")

        if label_list is None:
            label_list = [[label] for label in
                          self._substate_labels_list[cal_index]]

        state_labels = self._substate_labels_list[cal_index]
        fidelity_label_list = []
        if label_list is None:
            fidelity_label_list = [[label] for label in state_labels]
        else:
            for fid_sublist in label_list:
                fidelity_label_list.append([])
                for fid_statelabl in fid_sublist:
                    for label_idx, label in enumerate(state_labels):
                        if fid_statelabl == label:
                            fidelity_label_list[-1].append(label_idx)
                            continue

        # fidelity_label_list is a 2D list of indices in the
        # cal_matrix, we find the assignment fidelity of each
        # row and average over the list
        assign_fid_list = []

        for fid_label_sublist in fidelity_label_list:
            assign_fid_list.append(0)
            for state_idx_i in fid_label_sublist:
                for state_idx_j in fid_label_sublist:
                    assign_fid_list[-1] += \
                        self._cal_matrices[cal_index][state_idx_i][state_idx_j]
            assign_fid_list[-1] /= len(fid_label_sublist)

        return np.mean(assign_fid_list)

    def _build_calibration_matrices(self):
        """
        Build the measurement calibration matrices from the results of running
        the circuits returned by `measurement_calibration`.
        """

        # initialize the set of empty calibration matrices
        self._cal_matrices = []
        for list_size in self._qubit_list_sizes:
            self._cal_matrices.append(np.zeros([2**list_size, 2**list_size],
                                               dtype=float))

        # go through for each calibration experiment
        for result in self._result_list:
            for experiment in result.results:
                circ_name = experiment.header.name
                # extract the state from the circuit name
                # this was the prepared state
                circ_search = re.search('(?<=' + self._circlabel + 'cal_)\\w+',
                                        circ_name)

                # this experiment is not one of the calcs so skip
                if circ_search is None:
                    continue

                state = circ_search.group(0)

                # get the counts from the result
                state_cnts = result.get_counts(circ_name)
                for measured_state, counts in state_cnts.items():
                    end_index = self.nqubits
                    for cal_ind, cal_mat in enumerate(self._cal_matrices):

                        start_index = end_index - \
                            self._qubit_list_sizes[cal_ind]

                        substate_index = self._indices_list[cal_ind][
                            state[start_index:end_index]]
                        measured_substate_index = \
                            self._indices_list[cal_ind][
                                measured_state[start_index:end_index]]
                        end_index = start_index

                        cal_mat[measured_substate_index][substate_index] += \
                            counts

        for mat_index, _ in enumerate(self._cal_matrices):
            sums_of_columns = np.sum(self._cal_matrices[mat_index], axis=0)
            # pylint: disable=assignment-from-no-return
            self._cal_matrices[mat_index] = np.divide(
                self._cal_matrices[mat_index], sums_of_columns,
                out=np.zeros_like(self._cal_matrices[mat_index]),
                where=sums_of_columns != 0)

    def plot_calibration(self, cal_index=0, ax=None, show_plot=True):
        """
        Plot one of the calibration matrices (2D color grid plot).

        Args:
            cal_index(integer): calibration matrix to plot
            ax(matplotlib.axes): settings for the graph
            show_plot (bool): call plt.show()

        Raises:
            QiskitError: if _cal_matrices was not set.

            ImportError: if matplotlib was not installed.

        """

        if self._cal_matrices is None:
            raise QiskitError("Cal matrix has not been set")

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_rb_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            plt.figure()
            ax = plt.gca()

        axim = ax.matshow(self.cal_matrices[cal_index],
                          cmap=plt.cm.binary,
                          clim=[0, 1])
        ax.figure.colorbar(axim)
        ax.set_xlabel('Prepared State')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Measured State')
        ax.set_xticks(np.arange(len(self._substate_labels_list[cal_index])))
        ax.set_yticks(np.arange(len(self._substate_labels_list[cal_index])))
        ax.set_xticklabels(self._substate_labels_list[cal_index])
        ax.set_yticklabels(self._substate_labels_list[cal_index])

        if show_plot:
            plt.show()

#########################################################################################

def get_counts_result(output_correction, result_hpc, result_key: str, gauss_key: str, time_vector: list,
                      zne_extrapolation: bool, scale_factors: list, num_replicas: int, ignis: bool = False,
                      shots: int = 1000, meas_filter=None) -> Tuple[list, list]:
    results = list()
    experiments_params = get_exp_params(time_vector, zne_extrapolation, scale_factors, num_replicas)
    time_steps = len(time_vector)
    num_scales = len(scale_factors) if zne_extrapolation else 1

    for exp_ind in range(time_steps * num_scales * num_replicas):
        non_corrected_counts = result_hpc.get_counts(exp_ind)
        one_count = non_corrected_counts.get(result_key, 0) / shots
        experiment_result = dict()
        replica_ind = experiments_params[exp_ind][-1]

        experiment_result.update({
            'replica': replica_ind,
        })

        if zne_extrapolation:
            time_ind = experiments_params[exp_ind][0]
            scale_ind = experiments_params[exp_ind][1]
            experiment_result.update({
                'scale_factor': scale_ind,
            })
        else:
            time_ind = experiments_params[exp_ind][0]

        experiment_result.update({
            'time': time_ind
        })

        if ignis:
            corrected_counts = meas_filter.apply(non_corrected_counts)
            corrected_one_count = apply_ignis_error_correction(corrected_counts, exp_ind, result_key, shots)
            gauss_law_obs, gauss_law_obs_corrected = gauss_law(non_corrected_counts, corrected_counts, shots)
            sector_2_obs, sector_2_obs_corrected = sector_2(gauss_key, non_corrected_counts, corrected_counts,
                                                            shots)
            gauss_law_sq_obs, gauss_law_sq_obs_corrected = gauss_law_squared(non_corrected_counts, corrected_counts,
                                                                             shots)
            experiment_result.update({
                'gauss_law': gauss_law_obs,
                'gauss_law_corrected': gauss_law_obs_corrected,
                'sector_2': sector_2_obs,
                'sector_2_corrected': sector_2_obs_corrected,
                'gauss_law_squared': gauss_law_sq_obs,
                'gauss_law_squared_corrected': gauss_law_sq_obs_corrected,
            })
        else:
            corrected_one_count = apply_error_correction(non_corrected_counts, output_correction, result_key, shots)

        experiment_result.update({
            'original': one_count,
            'output_corrected': corrected_one_count,
        })

        results.append(experiment_result)

    results_df = pd.DataFrame(results)

    return results_df


def apply_error_correction(experiment_data, error_correction: dict, result_key: str, shots: int = 1000):
    if error_correction is None:
        return experiment_data.get(result_key, 0)

    possible_states = error_correction.get(STATES)
    row_to_choose = possible_states.index(result_key)
    correction_vector = error_correction.get(MATRIX)[row_to_choose]
    experiment_vector = [experiment_data.get(state, 0) / shots for state in possible_states]

    res = 0

    for exp_res, cor_val in zip(experiment_vector, correction_vector):
        res += exp_res * cor_val

    return res


def apply_ignis_error_correction(mitigated_counts, circ_ind: int, result_key: str, shots: int):
    return mitigated_counts.get(result_key, 0) / shots


class CustomErrorMitigation:
    def __init__(self, n_qubits: int = 4, shots: int = 1000):
        self.n_qubits = n_qubits
        self.shots = shots

    def _build_set_of_states(self):
        possible_states = list()
        for state in itertools.product([0, 1], repeat=self.n_qubits):
            state_to_str = list(map(str, state))
            possible_states.append(''.join(state_to_str))

        return possible_states

    def _get_probabilities_vector(self, counts: dict):
        possible_states = self._build_set_of_states()
        probability_vector = list()
        for state in possible_states:
            probability_vector.append(counts.get(state, 0) / self.shots)

        return probability_vector

    def _build_circuit(self, initial_state: str):
        if self.n_qubits != len(initial_state):
            raise Exception('Error in parameters, number of qubits does not agree with initial_state')
        q = QuantumRegister(self.n_qubits, 'q')
        circ = QuantumCircuit(q)
        for q_ind, q_state in enumerate(initial_state[::-1]):  # Flipping the state to map to qubit order
            if q_state == '1':
                circ.x(q[q_ind])

        c = ClassicalRegister(self.n_qubits, 'c')
        meas = QuantumCircuit(q, c)
        meas.measure(q, c)
        qc = circ + meas

        return qc

    def build_probability_matrix(self, backend: IBMBackend):
        possible_states = self._build_set_of_states()
        probability_matrix = list()
        circuits = list()
        for initial_state in possible_states:
            qc = self._build_circuit(initial_state)
            circuits.append(qc)

        # job_hpc = execute(circuits, backend=backend, shots=self.shots, max_credits=5)
        service = QiskitRuntimeService()

        # Submit the job using the "sampler" program
        job_hpc = service._run(
            program_id="sampler",  # Use "estimator" if needed
            options={"backend": backend.name},
            inputs={"circuits": circuits, "shots": self.shots}
        )

        result_hpc = job_hpc.result()

        for ind, _ in enumerate(possible_states):
            probability_matrix.append(self._get_probabilities_vector(result_hpc.get_counts(circuits[ind])))

        return {MATRIX: np.linalg.inv(probability_matrix), STATES: possible_states}

    @staticmethod
    def save_correction_results(error_correction: dict, filename: str):
        with open(filename, 'w') as file:
            json.dump(error_correction, file)

    @staticmethod
    def load_correction_results(filename: str):
        with open(filename, 'r') as file:
            error_correction = json.load(file)
        return error_correction


class IgnisErrorMitigation:
    def __init__(self, n_qubits: int = 4, shots: int = 1000):
        self.n_qubits = n_qubits
        self.shots = shots
        self.meas_fitter = None

    def get_meas_fitter(self, backend: IBMBackend):
        q_bits = list(range(self.n_qubits))
        cal_circuits, state_labels = complete_meas_cal(qubit_list=q_bits, circlabel='mitigationError')

        # cal_job = execute(cal_circuits,
        #                   backend=backend,
        #                   shots=self.shots,
        #                   optimization_level=0)
        service = QiskitRuntimeService()

        # Submit the job using the "sampler" program
        
        cal_job = service._run(
            program_id="sampler",  # Use "estimator" if needed
            options={"backend": backend.name},
            inputs={"circuits": cal_circuits, "shots": self.shots , "optimization_level":0}
        )

        cal_results = cal_job.result()
        meas_fitter = CompleteMeasFitter(cal_results, state_labels)
        sep_labs = list(map(lambda lab: " ".join(lab), meas_fitter.filter.state_labels))
        filter_obj = meas_fitter.filter
        filter_obj._state_labels = list(sep_labs)
        self.meas_fitter = meas_fitter
        return meas_fitter.filter

    def plot_calibration(self):
        self.meas_fitter.plot_calibration()


def get_exp_params(time_vector: list, zne_extrapolation: bool, scale_factors: list, num_replicas: int):
    replicas = list(range(num_replicas))
    if not zne_extrapolation:
        return np.array(np.meshgrid(time_vector, replicas)).T.reshape(-1, 2)

    return np.array(np.meshgrid(time_vector, scale_factors, replicas)).T.reshape(-1, 3)
