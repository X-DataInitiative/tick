//
//  interrupt.h
//  TICK
//
//  Created by bacry on 31/12/2015.
//  Copyright (c) 2015 bacry. All rights reserved.
//

#ifndef TICK_BASE_SRC_INTERRUPTION_H_
#define TICK_BASE_SRC_INTERRUPTION_H_

#include <atomic>
#include <exception>

/*! \class Interruption
 * \brief Exception Class made to handle Ctrl-C interruption
 */
class Interruption : public std::exception {
 private:
    static std::atomic<bool> flag_interrupt;

 public:
    //! @brief Simple constructor
    Interruption() {}

    const char *what() const noexcept override;

    //! \cond
    //! @brief Notifies an interruption has been detected (not to be called directly)
    inline static void set() { flag_interrupt = true; }

    //! @brief Reset interruption flag. Called when interruption has been processed
    //! (not to be called directly)
    inline static void reset() { flag_interrupt = false; }
    //! \endcond

    //! @brief Test whether Ctrl-C interruption has been detected
    //! \return true if interruption detected false otherwise
    inline static bool is_raised() { return flag_interrupt; }

    //! @brief Throw an exception (of type Interruption) if interruption has been detected
    //! \warning Never call it from inside a thread unless you use
    //! ::parallel_map or ::parallel_run
    inline static void throw_if_raised() { if (flag_interrupt) throw (Interruption()); }
};

#endif  // TICK_BASE_SRC_INTERRUPTION_H_
