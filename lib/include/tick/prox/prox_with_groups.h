#ifndef LIB_INCLUDE_TICK_PROX_PROX_WITH_GROUPS_H_
#define LIB_INCLUDE_TICK_PROX_PROX_WITH_GROUPS_H_

// License: BSD 3 clause

/**
 * Templating updates
 *  Since the addition of templating this file class has been
 *  required to be entirely in a header file.
 *  This is solely for the Windows MSVC. which complains
 *  about missing or deleted function for "std::unique_ptr"
 */

#include "prox.h"

template <class T, class K>
class TProxWithGroups : public TProx<T, K> {
 protected:
  using TProx<T, K>::has_range;
  using TProx<T, K>::strength;

 public:
  using TProx<T, K>::get_class_name;

 protected:
  // Tells us if the prox is ready (with correctly allocated sub-prox for each
  // blocks). This is mainly necessary when the user changes the range from
  // python
  bool is_synchronized = false;

  ulong n_blocks = 0;

  SArrayULongPtr blocks_start;
  SArrayULongPtr blocks_length;

  // A vector that contains the prox for each block
  std::vector<std::unique_ptr<TProx<T, K> > > proxs;

  void synchronize_proxs();

  virtual std::unique_ptr<TProx<T, K> > build_prox(T strength, ulong start,
                                                   ulong end, bool positive);

 protected:
  // This exists soley for cereal/swig
  TProxWithGroups() : TProxWithGroups(0, nullptr, nullptr, false) {}

 public:
  TProxWithGroups(T strength, SArrayULongPtr blocks_start,
                  SArrayULongPtr blocks_length, bool positive);

  TProxWithGroups(T strength, SArrayULongPtr blocks_start,
                  SArrayULongPtr blocks_length, ulong start, ulong end,
                  bool positive);

  // There's something odd on windows trying to copy the unique_ptr
  TProxWithGroups(const TProxWithGroups &) = delete;
  TProxWithGroups(const TProxWithGroups &&) = delete;
  TProxWithGroups &operator=(const TProxWithGroups &) = delete;
  TProxWithGroups &operator=(const TProxWithGroups &&) = delete;

  T value(const Array<K> &coeffs, ulong start, ulong end) override;

  void call(const Array<K> &coeffs, T step, Array<K> &out, ulong start,
            ulong end) override;

  inline void set_positive(bool positive) override {
    if (positive != this->positive) {
      is_synchronized = false;
    }
    this->positive = positive;
  }

  // We overload set_start_end here, since we'd need to update proxs when it's
  // changed
  inline void set_start_end(ulong start, ulong end) override {
    if ((start != this->start) || (end != this->end)) {
      // If we change the range, we need to update again the proxs
      is_synchronized = false;
    }
    this->has_range = true;
    this->start = start;
    this->end = end;
  }

  inline virtual void set_blocks_start(SArrayULongPtr blocks_start) {
    n_blocks = blocks_start->size();
    if (n_blocks != blocks_length->size()) {
      throw std::invalid_argument(
          "blocks_start and blocks_length must have the same size");
    }
    this->blocks_start = blocks_start;
    is_synchronized = false;
  }

  inline virtual void set_blocks_length(SArrayULongPtr blocks_length) {
    n_blocks = blocks_length->size();
    if (n_blocks != blocks_start->size()) {
      throw std::invalid_argument(
          "blocks_length and blocks_start must have the same size");
    }
    this->blocks_length = blocks_length;
    is_synchronized = false;
  }

 protected:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("Prox", cereal::base_class<TProx<T, K> >(this)));
    ar(CEREAL_NVP(is_synchronized));
    ar(CEREAL_NVP(n_blocks));
    ar(CEREAL_NVP(blocks_start));
    ar(CEREAL_NVP(blocks_length));
    ar(CEREAL_NVP(proxs));
  }

  BoolStrReport compare(const TProxWithGroups<T, K> &that,
                        std::stringstream &ss) {
    auto are_equal = TProx<T, K>::compare(that, ss) &&
                     this->proxs.size() == that.proxs.size();
    if (are_equal) {
      for (size_t i = 0; i < this->proxs.size(); i++) {
        are_equal = this->proxs[i] == that.proxs[i];
        if (!are_equal) break;
      }
    }
    return BoolStrReport(are_equal, ss.str());
  }
};

template <class T, class K>
TProxWithGroups<T, K>::TProxWithGroups(T strength, SArrayULongPtr blocks_start,
                                       SArrayULongPtr blocks_length,
                                       bool positive)
    : TProx<T, K>(strength, positive), is_synchronized(false) {
  this->blocks_start = blocks_start;
  this->blocks_length = blocks_length;
  // blocks_start and blocks_end have the same size
  if (!blocks_start) TICK_ERROR("ProxWithGroups blocks_start cannot be empty");
  n_blocks = blocks_start->size();
  // The object is not ready (synchronize_proxs has not been called)
  is_synchronized = false;
}

template <class T, class K>
TProxWithGroups<T, K>::TProxWithGroups(T strength, SArrayULongPtr blocks_start,
                                       SArrayULongPtr blocks_length,
                                       ulong start, ulong end, bool positive)
    : TProx<T, K>(strength, start, end, positive) {
  this->blocks_start = blocks_start;
  this->blocks_length = blocks_length;
  // blocks_start and blocks_end have the same size
  if (!blocks_start) TICK_ERROR("ProxWithGroups blocks_start cannot be empty");
  n_blocks = blocks_start->size();
  // The object is not ready (synchronize_proxs has not been called)
  is_synchronized = false;
}

template <class T, class K>
void TProxWithGroups<T, K>::synchronize_proxs() {
  proxs.clear();
  for (ulong k = 0; k < n_blocks; k++) {
    ulong start = (*blocks_start)[k];
    if (has_range) {
      // If there is a range, we apply the global start
      start += this->start;
    }
    ulong end = start + (*blocks_length)[k];
    proxs.emplace_back(build_prox(strength, start, end, this->positive));
  }
  is_synchronized = true;
}

template <class T, class K>
std::unique_ptr<TProx<T, K> > TProxWithGroups<T, K>::build_prox(T strength,
                                                                ulong start,
                                                                ulong end,
                                                                bool positive) {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

template <class T, class K>
T TProxWithGroups<T, K>::value(const Array<K> &coeffs, ulong start, ulong end) {
  if (!is_synchronized) {
    synchronize_proxs();
  }
  T val = 0.;
  for (auto &prox : proxs) {
    val += prox->value(coeffs, prox->get_start(), prox->get_end());
  }
  return val;
}

template <class T, class K>
void TProxWithGroups<T, K>::call(const Array<K> &coeffs, T step, Array<K> &out,
                                 ulong start, ulong end) {
  if (!is_synchronized) {
    synchronize_proxs();
  }
  for (auto &prox : proxs) {
    prox->call(coeffs, step, out, prox->get_start(), prox->get_end());
  }
}

#endif  // LIB_INCLUDE_TICK_PROX_PROX_WITH_GROUPS_H_
