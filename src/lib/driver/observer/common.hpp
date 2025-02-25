#pragma once

#include "core/types.hpp"

namespace dacti::observer {

	/// @brief Base class for all observers
	///	Note this class is abstract and should not be instantiated directly
	class Observer {
	protected:
		size_t _NumIntegVars;
		size_t _NumFieldVars;

		std::string _observer_name;
		std::vector<std::string> _IntegVar_names;
		std::vector<std::string> _FieldVar_names;

		mutable std::vector<scalar_t> _IntegVars;
		mutable std::vector<std::vector<scalar_t>> _FieldVars;

	public:
		Observer(std::string observer_name               = "Default Observer",
		         std::vector<std::string> IntegVar_names = {},
		         std::vector<std::string> FieldVar_names = {}) :
		    _observer_name(observer_name),
		    _IntegVar_names(IntegVar_names),
		    _FieldVar_names(FieldVar_names),
		    _NumIntegVars(IntegVar_names.size()),
		    _NumFieldVars(FieldVar_names.size()) {
			_IntegVars.resize(_NumIntegVars);
			_FieldVars.resize(_NumFieldVars);
		}

		size_t NumIntegVars() const { return _NumIntegVars; }
		size_t NumFieldVars() const { return _NumFieldVars; }
		std::string name() const { return _observer_name; }
		std::vector<std::string> IntegVar_names() const { return _IntegVar_names; }
		std::vector<std::string> FieldVar_names() const { return _FieldVar_names; }
		std::vector<scalar_t> IntegVars() const { return _IntegVars; }
		std::vector<std::vector<scalar_t>> FieldVars() const { return _FieldVars; }
	};
}    // namespace dacti::observer