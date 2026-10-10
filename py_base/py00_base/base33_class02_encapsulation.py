from __future__ import annotations

import copy

import pytest


class AccountError(Exception):
    pass

class InsufficientBalanceError(AccountError):
    pass
class AccountFrozenError(AccountError):
    pass

## 4. 阶段 2：封装与属性控制
### 4.1 练习四：银行账户
class BankAccount:
    def __init__(self, account_id: int, owner: str, balance: float, is_frozen: bool):
        """
            account_id: 账户编号
            owner: 账户持有人
            is_frozen: 账户是否冻结
        """
        self.account_id = account_id
        self.owner = owner
        self.is_frozen = is_frozen
        # _balance: 余额
        self._balance = balance
        # __transaction_history: 交易记录
        self.__transaction_history = []

        pass

    def deposit(self, amount: float) -> None:
        """ 存款 """
        self._do_deposit(amount)

        self.add_transaction_history("deposit", amount)

    def withdraw(self, amount: float):
        """ 取款 """
        self._do_withdraw(amount)

        self.add_transaction_history("withdrawal", amount)

    def _do_deposit(self, amount: float):
        # - 冻结账户不能存取款；
        self._validate_freeze()

        # - 存款金额必须大于 0；
        if amount < 0:
            raise AccountError(f"account_id={self.account_id}, msg=存款金额必须大于 0 ")

        self._balance += amount

    def _do_withdraw(self, amount: float):
        # - 冻结账户不能存取款；
        self._validate_freeze()

        # - 余额不足不能取款；
        if amount > self._balance:
            raise InsufficientBalanceError(f"account_id={self.account_id}, msg=余额不足 ")

        self._balance -= amount

    def _validate_freeze(self):
        """ 验证账户是否被冻结 """
        if self.is_frozen:
            raise AccountFrozenError(f"account_id={self.account_id}, msg=account was frozened ")

    def add_transaction_history(self, opt:str, amount: float):
        """ 新增交易历史记录 """
        item = {
            "opt": opt,
            "opt_amount": amount,
            "balance": self._balance,
        }
        self.__transaction_history.append(item)

    def freeze(self):
        """ 冻结账户 """
        self.is_frozen = True

    def unfreeze(self):
        """ 解冻账户 """
        self.is_frozen = False

    def get_transaction_history(self):
        """ 返回交易记录副本 """
        # - 外部代码不能直接替换交易历史列表；
        # - 返回交易历史时不能暴露内部可变列表；
        transaction_history = copy.deepcopy(self.__transaction_history)
        return transaction_history


    def transfer_to(self, dest_account: BankAccount, amount: float):
        # - 转账必须保证扣款和入账逻辑完整；
        process : str = ""
        try:
            # 先减后增
            self._do_withdraw(amount)
            process = "withdraw"
            dest_account._do_deposit(amount)
            process = "deposit"
        except AccountError as e:
            print(e)
        finally:
            if process == "withdraw":
                # 转账失败，把当前账户的钱给加回去
                self._do_deposit(amount)
            if process == "deposit":
                # 转账成功记录日志
                self.add_transaction_history("withdrawal", amount)
                dest_account.add_transaction_history("deposit", amount)

    @property
    def balance(self):
        """ 属性：只读获取余额。 """
        return self._balance

    pass



def create_account(
    account_id=10001,
    owner="Tom",
    balance=1000.0,
    is_frozen=False,
):
    return BankAccount(
        account_id=account_id,
        owner=owner,
        balance=balance,
        is_frozen=is_frozen,
    )


# ============================================================
# 1. 初始化
# ============================================================

def test_create_account():
    account = create_account()

    assert account.account_id == 10001
    assert account.owner == "Tom"
    assert account.balance == 1000.0
    assert account.is_frozen is False


def test_create_frozen_account():
    account = create_account(is_frozen=True)

    assert account.is_frozen is True
    assert account.balance == 1000.0


# ============================================================
# 2. balance 只读属性
# ============================================================

def test_balance_is_read_only():
    account = create_account(balance=1000)

    assert account.balance == 1000

    with pytest.raises(AttributeError):
        account.balance = 2000


def test_balance_can_be_read_after_deposit():
    account = create_account(balance=1000)

    account.deposit(500)

    assert account.balance == 1500


def test_balance_can_be_read_after_withdraw():
    account = create_account(balance=1000)

    account.withdraw(300)

    assert account.balance == 700


# ============================================================
# 3. deposit 存款
# ============================================================

def test_deposit():
    account = create_account(balance=1000)

    result = account.deposit(500)

    assert result is None
    assert account.balance == 1500


def test_deposit_zero():
    account = create_account(balance=1000)

    account.deposit(0)

    assert account.balance == 1000


def test_deposit_negative_amount():
    account = create_account(balance=1000)

    with pytest.raises(AccountError):
        account.deposit(-100)

    assert account.balance == 1000


# ============================================================
# 4. withdraw 取款
# ============================================================

def test_withdraw():
    account = create_account(balance=1000)

    result = account.withdraw(300)

    assert result is None
    assert account.balance == 700


def test_withdraw_all_balance():
    account = create_account(balance=1000)

    account.withdraw(1000)

    assert account.balance == 0


def test_withdraw_insufficient_balance():
    account = create_account(balance=1000)

    with pytest.raises(InsufficientBalanceError):
        account.withdraw(1001)

    assert account.balance == 1000


def test_withdraw_does_not_change_balance_when_failed():
    account = create_account(balance=1000)

    try:
        account.withdraw(2000)
    except InsufficientBalanceError:
        pass

    assert account.balance == 1000


# ============================================================
# 5. 冻结账户
# ============================================================

def test_freeze():
    account = create_account()

    assert account.is_frozen is False

    account.freeze()

    assert account.is_frozen is True


def test_unfreeze():
    account = create_account(is_frozen=True)

    assert account.is_frozen is True

    account.unfreeze()

    assert account.is_frozen is False


def test_frozen_account_cannot_deposit():
    account = create_account(
        balance=1000,
        is_frozen=True,
    )

    with pytest.raises(AccountFrozenError):
        account.deposit(500)

    assert account.balance == 1000


def test_frozen_account_cannot_withdraw():
    account = create_account(
        balance=1000,
        is_frozen=True,
    )

    with pytest.raises(AccountFrozenError):
        account.withdraw(500)

    assert account.balance == 1000


def test_unfreeze_then_can_deposit():
    account = create_account(
        balance=1000,
        is_frozen=True,
    )

    account.unfreeze()
    account.deposit(500)

    assert account.balance == 1500


def test_unfreeze_then_can_withdraw():
    account = create_account(
        balance=1000,
        is_frozen=True,
    )

    account.unfreeze()
    account.withdraw(500)

    assert account.balance == 500


# ============================================================
# 6. 交易记录
# ============================================================

def test_initial_transaction_history_is_empty():
    account = create_account()

    history = account.get_transaction_history()

    assert history == []


def test_deposit_creates_transaction_history():
    account = create_account(balance=1000)

    account.deposit(500)

    history = account.get_transaction_history()

    assert len(history) == 1

    assert history[0] == {
        "opt": "deposit",
        "opt_amount": 500,
        "balance": 1500,
    }


def test_withdraw_creates_transaction_history():
    account = create_account(balance=1000)

    account.withdraw(300)

    history = account.get_transaction_history()

    assert len(history) == 1

    assert history[0] == {
        "opt": "withdrawal",
        "opt_amount": 300,
        "balance": 700,
    }


def test_multiple_transactions():
    account = create_account(balance=1000)

    account.deposit(500)
    account.withdraw(300)
    account.deposit(200)

    history = account.get_transaction_history()

    assert len(history) == 3

    assert history[0] == {
        "opt": "deposit",
        "opt_amount": 500,
        "balance": 1500,
    }

    assert history[1] == {
        "opt": "withdrawal",
        "opt_amount": 300,
        "balance": 1200,
    }

    assert history[2] == {
        "opt": "deposit",
        "opt_amount": 200,
        "balance": 1400,
    }


# ============================================================
# 7. 交易记录不能暴露内部列表
# ============================================================

def test_transaction_history_is_deep_copy():
    account = create_account(balance=1000)

    account.deposit(500)

    history = account.get_transaction_history()

    history.append({
        "opt": "fake",
        "opt_amount": 999999,
        "balance": 999999,
    })

    # 修改返回值不能影响内部交易记录
    new_history = account.get_transaction_history()

    assert len(new_history) == 1
    assert new_history[0]["opt"] == "deposit"


def test_transaction_history_nested_data_is_copy():
    account = create_account(balance=1000)

    account.deposit(500)

    history = account.get_transaction_history()

    history[0]["balance"] = 999999

    new_history = account.get_transaction_history()

    assert new_history[0]["balance"] == 1500


# ============================================================
# 8. 转账
# ============================================================

def test_transfer_success():
    source = create_account(
        account_id=10001,
        balance=1000,
    )

    dest = create_account(
        account_id=10002,
        balance=500,
    )

    source.transfer_to(dest, 300)

    assert source.balance == 700
    assert dest.balance == 800


def test_transfer_source_transaction_history():
    source = create_account(
        account_id=10001,
        balance=1000,
    )

    dest = create_account(
        account_id=10002,
        balance=500,
    )

    source.transfer_to(dest, 300)

    history = source.get_transaction_history()

    assert len(history) == 1

    assert history[0] == {
        "opt": "withdrawal",
        "opt_amount": 300,
        "balance": 700,
    }


def test_transfer_destination_transaction_history():
    source = create_account(
        account_id=10001,
        balance=1000,
    )

    dest = create_account(
        account_id=10002,
        balance=500,
    )

    source.transfer_to(dest, 300)

    history = dest.get_transaction_history()

    assert len(history) == 1

    assert history[0] == {
        "opt": "deposit",
        "opt_amount": 300,
        "balance": 800,
    }


# ============================================================
# 9. 转账：余额不足
# ============================================================

def test_transfer_insufficient_balance():
    source = create_account(
        account_id=10001,
        balance=100,
    )

    dest = create_account(
        account_id=10002,
        balance=500,
    )

    source.transfer_to(dest, 300)

    # 转账失败，双方余额都应该保持不变
    assert source.balance == 100
    assert dest.balance == 500


def test_transfer_insufficient_balance_history():
    source = create_account(
        account_id=10001,
        balance=100,
    )

    dest = create_account(
        account_id=10002,
        balance=500,
    )

    source.transfer_to(dest, 300)

    assert source.get_transaction_history() == []
    assert dest.get_transaction_history() == []


# ============================================================
# 10. 转账：目标账户被冻结
# ============================================================

def test_transfer_to_frozen_account():
    source = create_account(
        account_id=10001,
        balance=1000,
    )

    dest = create_account(
        account_id=10002,
        balance=500,
        is_frozen=True,
    )

    source.transfer_to(dest, 300)

    # 目标账户冻结，转账应该失败
    assert source.balance == 1000
    assert dest.balance == 500


def test_transfer_to_frozen_account_history():
    source = create_account(
        account_id=10001,
        balance=1000,
    )

    dest = create_account(
        account_id=10002,
        balance=500,
        is_frozen=True,
    )

    source.transfer_to(dest, 300)

    # 失败的转账不应该产生最终交易记录
    assert source.get_transaction_history() == []
    assert dest.get_transaction_history() == []


# ============================================================
# 11. 自定义异常类型
# ============================================================

def test_account_frozen_error_is_account_error():
    assert issubclass(AccountFrozenError, AccountError)


def test_insufficient_balance_error_is_account_error():
    assert issubclass(
        InsufficientBalanceError,
        AccountError,
    )

