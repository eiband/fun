/*
Copyright (c) 2019 Daniel Eiband

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <cassert>
#include <exception>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

/*
 * With or without synchronization?
 */
#define YOLO_SINGLE_THREADED

#if defined(YOLO_SINGLE_THREADED)
#define YOLO_NOEXCEPT noexcept
#else
#define YOLO_NOEXCEPT
#endif

#if !defined(YOLO_SINGLE_THREADED)
#include <mutex>
#endif

namespace yolo
{
  template <typename T>
  class future;

  namespace detail
  {
    template <typename T>
    struct is_future : std::false_type
    {
    };

    template <typename T>
    struct is_future<future<T>> : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_future_v = is_future<T>::value;

    template <typename T>
    struct future_unwrap
    {
      using type = T;
    };

    template <typename T>
    struct future_unwrap<future<T>>
    {
      using type = T;
    };

    template <typename T>
    using future_unwrap_t = typename future_unwrap<T>::type;

  } // namespace detail

  class future_error : public std::logic_error
  {
  public:
    explicit future_error(const char* str)
      : std::logic_error(str)
    {
    }
  };

  namespace detail
  {
    struct future_helper;

    /*
     * Maybe put the implementation into a cpp file to generate less code
     */
    [[noreturn]] void throw_future_error(const char* what)
    {
      throw future_error(what);
    }

    /*
     * Maybe put the implementation into a cpp file to generate less code
     */
    [[nodiscard]] std::exception_ptr make_future_error(const char* what)
    {
      return std::make_exception_ptr(future_error(what));
    }

    struct future_void
    {
    };

    template <typename T>
    using future_storage_t = std::conditional_t<std::is_void_v<T>, future_void, T>;
    template <typename T>
    using future_value = std::variant<std::monostate, future_storage_t<T>, std::exception_ptr>;

    template <typename T>
    struct is_valid_future_value
      : std::negation<std::disjunction<std::is_same<T, std::monostate>, std::is_same<T, std::exception_ptr>>>
    {
    };

    template <typename T>
    inline constexpr bool is_valid_future_value_v = is_valid_future_value<T>::value;

    template <typename T>
    struct is_valid_future_result
      : std::negation<
          std::disjunction<std::is_convertible<T, std::monostate>, std::is_convertible<T, std::exception_ptr>>>
    {
    };

    template <typename T>
    inline constexpr bool is_valid_future_result_v = is_valid_future_result<T>::value;

    struct future_state_base
    {
    };

    struct future_continuation;

    using future_next_ptr = std::unique_ptr<future_continuation>;
    using future_next = std::pair<future_next_ptr, std::shared_ptr<future_state_base>>;

    struct future_continuation
    {
      future_continuation() = default;
      virtual ~future_continuation() = default;

      future_continuation(const future_continuation& that) = delete;
      future_continuation& operator=(const future_continuation& that) = delete;

      [[nodiscard]] virtual future_next continue_with(future_state_base& state) = 0;
    };

    void execute_future(future_next next)
    {
      while (next.first)
        next = next.first->continue_with(*next.second);
    }

    template <typename T>
    struct future_state : public future_state_base
    {
      future_state() = default;

      future_state(const future_state& that) = delete;
      future_state& operator=(const future_state& that) = delete;

      [[nodiscard]] bool ready() const YOLO_NOEXCEPT
      {
#if !defined(YOLO_SINGLE_THREADED)
        const std::lock_guard lock(_mutex);
#endif

        return ready_impl();
      }

      [[nodiscard]] future_next_ptr next() noexcept
      {
#if !defined(YOLO_SINGLE_THREADED)
        const std::lock_guard lock(_mutex);
#endif

        assert(ready_impl());

        return std::move(_continuation);
      }

      [[nodiscard]] future_next_ptr chain(future_next_ptr&& cont) noexcept
      {
#if !defined(YOLO_SINGLE_THREADED)
        const std::lock_guard lock(_mutex);
#endif

        assert(!_continuation);

        if (ready_impl())
          return std::move(cont);

        _continuation = std::move(cont);

        return nullptr;
      }

      template <typename Arg>
      void set_value(Arg&& value)
      {
#if !defined(YOLO_SINGLE_THREADED)
        const std::lock_guard lock(_mutex);
#endif

        _value = std::forward<Arg>(value);
      }

      template <typename Arg>
      void set_value_unsafe(Arg&& value)
      {
        _value = std::forward<Arg>(value);
      }

      [[nodiscard]] future_value<T>&& move_value() noexcept
      {
        return std::move(_value);
      }

    private:
#if !defined(YOLO_SINGLE_THREADED)
      mutable std::mutex _mutex;
#endif

      future_value<T> _value;
      future_next_ptr _continuation;

      [[nodiscard]] bool ready_impl() const noexcept
      {
        return (_value.index() != 0);
      }
    };

    template <typename T, typename U>
    struct future_attach : future_continuation
    {
      explicit future_attach(std::shared_ptr<future_state<U>>&& dest)
        : future_continuation{}
        , _dest(std::move(dest))
      {
      }

      [[nodiscard]] future_next continue_with(future_state_base& state) override
      {
        _dest->set_value(static_cast<future_state<T>&>(state).move_value());

        future_next_ptr next = _dest->next();

        return {std::move(next), std::move(_dest)};
      }

    private:
      std::shared_ptr<future_state<U>> _dest;
    };

    template <typename T, typename U>
    [[nodiscard]] future_next attach_future(
      std::shared_ptr<future_state<T>>&& src,
      std::shared_ptr<future_state<U>>&& dest)
    {
      if (src)
      {
        if (!src->ready())
        {
          future_next_ptr next = src->chain(std::make_unique<future_attach<T, U>>(std::move(dest)));

          return {std::move(next), std::move(src)};
        }

        dest->set_value(src->move_value());
      }
      else
        dest->set_value(make_future_error("invalid future"));

      future_next_ptr next = dest->next();

      return {std::move(next), std::move(dest)};
    }

    template <typename T, typename U, typename Func>
    void invoke_future_then(future_state<T>& dest, future_value<U>&& src, Func&& func)
    {
      if constexpr (std::is_void_v<T>)
      {
        if constexpr (std::is_void_v<U>)
          std::invoke(std::forward<Func>(func));
        else
          std::invoke(std::forward<Func>(func), std::get<U>(std::move(src)));

        dest.set_value(future_void{});
      }
      else
      {
        if constexpr (std::is_void_v<U>)
          dest.set_value(std::invoke(std::forward<Func>(func)));
        else
          dest.set_value(std::invoke(std::forward<Func>(func), std::get<U>(std::move(src))));
      }
    }

    template <typename T, typename Func>
    struct future_invoke_result
    {
      using type = std::decay_t<std::invoke_result_t<Func, T>>;
    };

    template <typename Func>
    struct future_invoke_result<void, Func>
    {
      using type = std::decay_t<std::invoke_result_t<Func>>;
    };

    template <typename T, typename Func>
    using future_invoke_result_t = typename future_invoke_result<T, Func>::type;

    template <typename T, typename Func>
    using future_then_result_t = future_unwrap_t<future_invoke_result_t<T, Func>>;

    template <typename T, typename Func>
    struct future_then : future_continuation
    {
      using result_type = future_then_result_t<T, Func>;

      static_assert(
        detail::is_valid_future_result_v<result_type>,
        "T must not be convertible to any of the types used internally.");

      template <typename Arg>
      future_then(Arg&& func, std::shared_ptr<future_state<result_type>> state)
        : future_continuation{}
        , _func(std::forward<Arg>(func))
        , _state(std::move(state))
      {
      }

      [[nodiscard]] future_next continue_with(future_state_base& state) override
      {
        future_value<T>&& value = static_cast<future_state<T>&>(state).move_value();

        if (value.index() == 1)
        {
          try
          {
            if constexpr (is_future_v<future_invoke_result_t<T, Func>>)
            {
              auto fut = std::invoke(std::move(_func), std::get<T>(std::move(value)));

              return attach_future(std::move(fut._state), std::move(_state));
            }
            else
            {
              invoke_future_then<result_type, T>(*_state, std::move(value), std::move(_func));
            }
          }
          catch (...)
          {
            _state->set_value(std::current_exception());
          }
        }
        else
        {
          _state->set_value(std::get<std::exception_ptr>(std::move(value)));
        }

        future_next_ptr next = _state->next();

        return {std::move(next), std::move(_state)};
      }

    private:
      Func _func;
      std::shared_ptr<future_state<result_type>> _state;
    };

    template <typename T, typename Func>
    void invoke_future_catch(future_state<T>& dest, std::exception_ptr&& ex, Func&& func)
    {
      if constexpr (std::is_void_v<T>)
      {
        std::invoke(std::forward<Func>(func), std::move(ex));

        dest.set_value(future_void{});
      }
      else
      {
        dest.set_value(std::invoke(std::forward<Func>(func), std::move(ex)));
      }
    }

    template <typename Func>
    using future_catch_invoke_result_t = std::decay_t<std::invoke_result_t<Func, std::exception_ptr>>;

    template <typename T, typename Func>
    using future_catch_result_t = std::common_type_t<T, future_unwrap_t<future_catch_invoke_result_t<Func>>>;

    template <typename T, typename Func>
    struct future_catch : future_continuation
    {
      using result_type = future_catch_result_t<T, Func>;

      static_assert(
        std::disjunction_v<std::is_void<result_type>, std::is_convertible<T, result_type>>,
        "The result of the continuation function is not convertible to the future result type.");

      static_assert(
        detail::is_valid_future_result_v<result_type>,
        "T must not be convertible to any of the types used internally.");

      template <typename Arg>
      future_catch(Arg&& func, std::shared_ptr<future_state<result_type>> state)
        : future_continuation{}
        , _func(std::forward<Arg>(func))
        , _state(std::move(state))
      {
      }

      [[nodiscard]] future_next continue_with(future_state_base& state) override
      {
        future_value<T>&& value = static_cast<future_state<T>&>(state).move_value();

        if (value.index() == 1)
        {
          _state->set_value(std::move(value));
        }
        else
        {
          try
          {
            if constexpr (is_future_v<future_catch_invoke_result_t<Func>>)
            {
              auto fut = std::invoke(std::move(_func), std::get<std::exception_ptr>(std::move(value)));

              return attach_future(std::move(fut._state), std::move(_state));
            }
            else
            {
              invoke_future_catch<result_type>(
                *_state, std::get<std::exception_ptr>(std::move(value)), std::move(_func));
            }
          }
          catch (...)
          {
            _state->set_value(std::current_exception());
          }
        }

        future_next_ptr next = _state->next();

        return {std::move(next), std::move(_state)};
      }

    private:
      Func _func;
      std::shared_ptr<future_state<result_type>> _state;
    };

  } // namespace detail

  template <typename T>
  class future
  {
    friend detail::future_helper;

    static_assert(detail::is_valid_future_value_v<T>, "T must not be any of the types used internally.");

  public:
    future() = default;
    future(const future& that) = delete;
    future(future&& that) = default;

    future& operator=(const future& that) = delete;
    future& operator=(future&& that) = default;

    [[nodiscard]] bool valid() const noexcept
    {
      return (_state != nullptr);
    }

    [[nodiscard]] bool ready() const YOLO_NOEXCEPT
    {
      return _state && _state->ready();
    }

    template <typename Func>
    future<detail::future_then_result_t<T, std::decay_t<Func>>> then(Func&& func)
    {
      using result_type = detail::future_then_result_t<T, std::decay_t<Func>>;
      using continuation_type = detail::future_then<T, std::decay_t<Func>>;

      check();

      future<result_type> fut;
      fut._state = std::make_shared<detail::future_state<result_type>>();

      detail::future_next_ptr next =
        _state->chain(std::make_unique<continuation_type>(std::forward<Func>(func), fut._state));

      detail::execute_future({std::move(next), std::move(_state)});

      return fut;
    }

    template <typename Func>
    future<detail::future_catch_result_t<T, std::decay_t<Func>>> catch_exception(Func&& func)
    {
      using result_type = detail::future_catch_result_t<T, std::decay_t<Func>>;
      using continuation_type = detail::future_catch<T, std::decay_t<Func>>;

      check();

      future<result_type> fut;
      fut._state = std::make_shared<detail::future_state<result_type>>();

      detail::future_next_ptr next =
        _state->chain(std::make_unique<continuation_type>(std::forward<Func>(func), fut._state));

      detail::execute_future({std::move(next), std::move(_state)});

      return fut;
    }

  private:
    template <typename U>
    friend class future;
    template <typename U, typename Func>
    friend struct detail::future_then;
    template <typename U, typename Func>
    friend struct detail::future_catch;

    std::shared_ptr<detail::future_state<T>> _state;

    void check() const
    {
      if (!_state)
        detail::throw_future_error("invalid future");
    }
  };

  namespace detail
  {
    template <typename T>
    struct promise_base
    {
      std::shared_ptr<detail::future_state<T>> _state;

      promise_base() = default;
      promise_base(const promise_base& that) = delete;
      promise_base(promise_base&& that) = default;

      promise_base& operator=(const promise_base& that) = delete;
      promise_base& operator=(promise_base&& that) = default;

      ~promise_base()
      {
        if (_state)
          satisfy(make_future_error("broken promise"));
      }

      void check() const
      {
        if (!_state)
          throw_future_error("promise already satisfied");
      }

      template <typename Arg>
      void satisfy(Arg&& arg)
      {
        _state->set_value(std::forward<Arg>(arg));

        future_next_ptr next = _state->next();

        execute_future({std::move(next), std::move(_state)});
      }
    };

  } // namespace detail

  template <typename T>
  class promise : private detail::promise_base<T>
  {
    friend detail::future_helper;

    static_assert(detail::is_valid_future_value_v<T>, "T must not be any of the types used internally.");

  public:
    promise() = default;
    promise(const promise& that) = delete;
    promise(promise&& that) = default;

    promise& operator=(const promise& that) = delete;
    promise& operator=(promise&& that) = default;

    void set_value(const T& value)
    {
      this->check();
      this->satisfy(value);
    }

    void set_value(T&& value)
    {
      this->check();
      this->satisfy(std::move(value));
    }

    void set_exception(const std::exception_ptr& ex)
    {
      this->check();
      this->satisfy(ex);
    }

    void set_exception(std::exception_ptr&& ex)
    {
      this->check();
      this->satisfy(std::move(ex));
    }
  };

  template <>
  class promise<void> : private detail::promise_base<void>
  {
    friend detail::future_helper;

  public:
    promise() = default;
    promise(const promise& that) = delete;
    promise(promise&& that) = default;

    promise& operator=(const promise& that) = delete;
    promise& operator=(promise&& that) = default;

    void set_value()
    {
      this->check();
      this->satisfy(detail::future_void{});
    }

    void set_exception(const std::exception_ptr& ex)
    {
      this->check();
      this->satisfy(ex);
    }

    void set_exception(std::exception_ptr&& ex)
    {
      this->check();
      this->satisfy(std::move(ex));
    }
  };

  namespace detail
  {
    struct future_helper
    {
      template <typename T>
      static std::pair<promise<T>, future<T>> make()
      {
        promise<T> prm;
        future<T> fut;

        prm._state = fut._state = std::make_shared<future_state<T>>();

        return {std::move(prm), std::move(fut)};
      }

      template <typename T, typename Arg>
      static future<T> make_ready(Arg&& arg)
      {
        future<T> fut;

        fut._state = std::make_shared<future_state<T>>();
        fut._state->set_value_unsafe(std::forward<Arg>(arg));

        return fut;
      }
    };

  } // namespace detail

  template <typename T>
  [[nodiscard]] std::pair<promise<T>, future<T>> make_promise()
  {
    return detail::future_helper::make<T>();
  }

  template <typename T>
  [[nodiscard]] future<std::decay_t<T>> make_ready_future(T&& value)
  {
    return detail::future_helper::make_ready<std::decay_t<T>>(std::forward<T>(value));
  }

  template <typename T, typename Arg>
  [[nodiscard]] future<T> make_ready_future(Arg&& value)
  {
    return detail::future_helper::make_ready<T>(std::forward<Arg>(value));
  }

  template <typename T>
  [[nodiscard]] future<T> make_exceptional_future(const std::exception_ptr& ex)
  {
    return detail::future_helper::make_ready<T>(ex);
  }

  template <typename T>
  [[nodiscard]] future<T> make_exceptional_future(std::exception_ptr&& ex)
  {
    return detail::future_helper::make_ready<T>(std::move(ex));
  }

} // namespace yolo

int main()
{
  using namespace yolo;

  struct test_exception
  {
  };

  const auto throw_exception = []() -> long { throw test_exception{}; };

  const auto exception_to_five = [](std::exception_ptr ex) {
    try
    {
      std::rethrow_exception(std::move(ex));
    }
    catch (const test_exception&)
    {
      return 5;
    }
    catch (...)
    {
      return -1;
    }
  };

  // Combinations of int and void
  {
    auto [prm, fut] = make_promise<int>();

    assert(fut.valid() && !fut.ready());

    prm.set_value(5);

    assert(fut.valid() && fut.ready());

    int result = -1;
    fut.then([&result](int i) { result = i; });

    assert(!fut.valid() && (result == 5));
  }
  {
    auto [prm, fut] = make_promise<int>();

    assert(!fut.ready());

    int result = -1;
    fut.then([&result](int i) { result = i; });

    assert(!fut.valid() && (result == -1));

    prm.set_value(5);

    assert(result == 5);
  }
  {
    auto [prm, fut] = make_promise<void>();

    assert(fut.valid() && !fut.ready());

    prm.set_value();

    assert(fut.valid() && fut.ready());

    int result = -1;
    fut.then([&result]() { result = 5; });

    assert(!fut.valid() && (result == 5));
  }
  {
    auto [prm, fut] = make_promise<void>();

    assert(fut.valid() && !fut.ready());

    prm.set_value();

    assert(fut.valid() && fut.ready());

    int result = -1;
    fut.then([]() { return 5; }).then([&result](int i) { result = i; });

    assert(!fut.valid() && (result == 5));
  }
  {
    auto [prm, fut] = make_promise<void>();

    int result = -1;
    fut.then([]() { return 5; }).then([](int i) { return 2 * i; }).then([&result](int i) { result = i; });

    assert(!fut.valid() && (result == -1));

    prm.set_value();

    assert(result == 10);
  }

  // Error propagation
  {
    auto [prm, fut] = make_promise<void>();

    assert(fut.valid() && !fut.ready());

    int result = 5;
    fut.then([&result]() { result = -1; });

    prm.set_exception(std::make_exception_ptr(test_exception{}));

    assert(!fut.valid() && (result == 5));
  }
  {
    auto [prm, fut] = make_promise<void>();

    assert(fut.valid() && !fut.ready());

    prm.set_exception(std::make_exception_ptr(test_exception{}));

    int result = 5;
    fut.then([&result]() { result = -1; });

    assert(!fut.valid() && (result == 5));
  }
  {
    auto [prm, fut] = make_promise<long>();

    assert(fut.valid() && !fut.ready());

    long result = -1;
    fut.catch_exception(exception_to_five).then([&result](long l) { result = l; });

    prm.set_exception(std::make_exception_ptr(test_exception{}));

    assert(!fut.valid() && (result == 5));
  }
  {
    auto [prm, fut] = make_promise<long>();

    assert(fut.valid() && !fut.ready());

    long result = -1;
    fut.catch_exception(exception_to_five).then([&result](long l) { result = l; });

    prm.set_value(10);

    assert(!fut.valid() && (result == 10));
  }
  {
    auto [prm, fut] = make_promise<void>();

    assert(fut.valid() && !fut.ready());

    int result = -1;
    fut.catch_exception([=](std::exception_ptr) {}).then([&result]() { result = 5; });

    prm.set_exception(std::make_exception_ptr(test_exception{}));

    assert(!fut.valid() && (result == 5));
  }
  {
    auto [prm, fut] = make_promise<void>();

    bool called = false;
    const auto skipped = [&called](long l) {
      called = true;
      return l;
    };

    int result = -1;
    fut.then(throw_exception).then(skipped).catch_exception(exception_to_five).then([&result](long l) { result = l; });

    prm.set_value();

    assert(!fut.valid() && !called && (result == 5));
  }

  // Inner future
  {
    auto [prm0, fut0] = make_promise<int>();
    auto [prm1, fut1] = make_promise<std::unique_ptr<int>>();

    future<std::unique_ptr<long>> fut2 = fut0.then([fut = std::move(fut1)](int i) mutable {
      return fut.then([i](std::unique_ptr<int> pi) { return std::make_unique<long>(i * (pi ? *pi : 0)); });
    });

    long result = -1;
    fut2.then([&result](std::unique_ptr<long> l) { result = l ? *l : -1; });

    assert(!fut0.valid() && !fut1.valid() && !fut2.valid() && (result == -1));

    prm0.set_value(5);

    assert(result == -1);

    prm1.set_value(std::make_unique<int>(3));

    assert(result == 15);
  }
  {
    auto [prm0, fut0] = make_promise<int>();
    auto [prm1, fut1] = make_promise<std::unique_ptr<int>>();

    future<std::unique_ptr<long>> fut2 = fut0.then([fut = std::move(fut1)](int i) mutable {
      return fut.then([i](std::unique_ptr<int> pi) { return std::make_unique<long>(i * (pi ? *pi : 0)); });
    });

    long result = -1;
    fut2.then([&result](std::unique_ptr<long> l) { result = l ? *l : -1; });

    assert(!fut0.valid() && !fut1.valid() && !fut2.valid() && (result == -1));

    prm1.set_value(std::make_unique<int>(3));

    assert(result == -1);

    prm0.set_value(5);

    assert(result == 15);
  }
  {
    auto [prm0, fut0] = make_promise<std::unique_ptr<long>>();
    auto [prm1, fut1] = make_promise<std::unique_ptr<int>>();

    future<std::unique_ptr<long>> fut2 =
      fut0.catch_exception([=, fut = std::move(fut1)](std::exception_ptr ex) mutable {
        return fut.then([i = exception_to_five(std::move(ex))](std::unique_ptr<int> pi) {
          return std::make_unique<long>(i * (pi ? *pi : 0));
        });
      });

    long result = -1;
    fut2.then([&result](std::unique_ptr<long> l) { result = l ? *l : -1; });

    assert(!fut0.valid() && !fut1.valid() && !fut2.valid() && (result == -1));

    prm0.set_exception(std::make_exception_ptr(test_exception{}));

    assert(result == -1);

    prm1.set_value(std::make_unique<int>(3));

    assert(result == 15);
  }

  // Ready future
  {
    future<int> fut = make_ready_future(5);

    assert(fut.ready());

    int result = -1;
    fut.then([&result](int i) { result = i; });

    assert(!fut.valid() && (result == 5));
  }
  {
    future<long> fut = make_ready_future<long>(5);

    assert(fut.ready());

    long result = -1;
    fut.then([&result](long l) { result = l; });

    assert(!fut.valid() && (result == 5));
  }
  {
    future<int> fut = make_exceptional_future<int>(std::make_exception_ptr(test_exception{}));

    assert(fut.ready());

    long result = -1;
    fut.catch_exception(exception_to_five).then([&result](int i) { result = i; });

    assert(!fut.valid() && (result == 5));
  }

  return 0;
}
