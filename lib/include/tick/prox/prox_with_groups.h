#ifndef LIB_INCLUDE_TICK_PROX_PROX_WITH_GROUPS_H_
#define LIB_INCLUDE_TICK_PROX_PROX_WITH_GROUPS_H_

// License: BSD 3 clause

#include "prox.h"

class ProxWithGroups : public Prox {
 protected:
  bool positive;
  ulong n_blocks;

  SArrayULongPtr blocks_start;
  SArrayULongPtr blocks_length;

  // A vector that contains the prox for each block
  std::vector <std::unique_ptr<Prox>> proxs;

  // Tells us if the prox is ready (with correctly allocated sub-prox for each blocks).
  // This is mainly necessary when the user changes the range from python
  bool is_synchronized;

  void synchronize_proxs();

  virtual std::unique_ptr<Prox> build_prox(double strength, ulong start, ulong end, bool positive);

 public:
  ProxWithGroups(double strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                 bool positive);

  ProxWithGroups(double strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                 ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

  double value(const ArrayDouble &coeffs, ulong start, ulong end) override;

  void call(const ArrayDouble &coeffs, double step, ArrayDouble &out,
            ulong start, ulong end) override;

  inline void set_positive(bool positive) override {
    if (positive != this->positive) {
      is_synchronized = false;
    }
    this->positive = positive;
  }

  // We overload set_start_end here, since we'd need to update proxs when it's changed
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
      throw std::invalid_argument("blocks_start and blocks_length must have the same size");
    }
    this->blocks_start = blocks_start;
    is_synchronized = false;
  }

  inline virtual void set_blocks_length(SArrayULongPtr blocks_length) {
    n_blocks = blocks_length->size();
    if (n_blocks != blocks_start->size()) {
      throw std::invalid_argument("blocks_length and blocks_start must have the same size");
    }
    this->blocks_length = blocks_length;
    is_synchronized = false;
  }
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_WITH_GROUPS_H_
