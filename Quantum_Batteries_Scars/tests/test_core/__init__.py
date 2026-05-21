"""Tests for Hamiltonian implementations."""

import pytest
import numpy as np
from quantum_battery.core import Hamiltonian


class TestHamiltonianInterface:
    """Test the Hamiltonian base class interface."""
    
    def test_abstract_get_matrix(self):
        """Test that abstract get_matrix raises NotImplementedError."""
        h = Hamiltonian(dim=2)
        with pytest.raises(NotImplementedError):
            h.get_matrix()
    
    def test_hamiltonian_info(self):
        """Test getting Hamiltonian info."""
        h = Hamiltonian(dim=2, time_dependent=False)
        info = h.get_info()
        
        assert info["dimension"] == 2
        assert info["time_dependent"] is False
