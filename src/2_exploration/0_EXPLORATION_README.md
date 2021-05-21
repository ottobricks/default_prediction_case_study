# Profiling accounts known to have defaulted

Let's take advantage of the variable groups we created during sanity to guide our exploration.
Due to time constraints for the case study, we won't explore all variables in depth.
Rather, we will propose 7 research questions (RQ) to guide our exploration. 

The first 2 RQs come from variables that have no missing (or zero) values, the personal variables.
1. Are customers who default more likely to be younger?
2. Is there a relationship between how people choose their email and the probability of default?

The remaining 4 RQs come each from one of the variable groups we define during our sanity check:
3. Can knowing the status of previous orders distinguish the probability of default?
4. Is there a relationship between active (paid and still-paying) invoices and the probability of default?
5. blu
6. ble

For reference, the variable groups are:

**1. Personal variables**
- age
- name_in_email: how much of the account owner's name can be found in their email

**2. Status variables**<br/>
They seem to be ordinal (hence the max in some of them), with 1.0 being the normal status and 4.0 being the worst. From the principle "customer obssession", it seems reasonable to assume missing values should default to 1.0. In other words, the account is considered normal until proven otherwise.
- account_status: the current status of the account
- account_worst_status_M_Nm: the worst status (max) in the M to N period in months
- account_days_in_dc_M_Nm: number of days an account was put into status "dc" between M and N months ago (?)
- account_days_in_rem_M_Nm: number of days an account was put into status "rem" between M and N months ago (?)
- account_days_in_term_M_Nm: number of days an account was put into status "term" between M and N months ago (terminated ?)
- status_max_archived_M_N_months: the max (worst) status for archieved invoices
- status_Nth_archived_M_Nm: status of the last Nth archived invoice between M and N months ago
    - 0 through 5, with 1 being the most commom status... perhaps:
        - 0 means paid on the spot
        - 1 means paid fully in short term installments
        - 2 means paid fully in longer term installments
        - 3 means paid fully in installments with fee due to delays in payment
        - 4 means not paid in full (is not present in the train dataset)
        - 5 means not paid at all
    
**3. Account variables**
- account_amount_added_12_24m: money deposited into the app for cardless purchases
- sum_capital_paid_account_M_Nm: the total amount of money paid with account funds between M and N months ago
- account_incoming_debt_vs_paid_0_24m: ratio "new_debt / old_paid_debt"
- num_unpaid_bills: number of installments still unpaid
- sum_paid_inv_0_12m: the total amount of money paid with installments (card) between M and N months ago
- max_paid_inv_M_Nm: the amount of the largest purchase the account has made between M and N months ago
- num_active_inv: number of active invoices
- num_active_div_by_paid_inv_M_Nm: ratio "number_active_invoices / number_paid_invoices" between M and N months ago
    - null when number_paid_invoices is zero
    - can be imputed, what's the best way?
- avg_payment_span_M_Nm: the average number of days to liquidate debt looking back between M and N months ago
- has_paid: whether the account has paid the latest installment

**4. Archived variables**
- num_arch_dc_M_Nm: number of archieved invoices with status "dc" between M and N months ago (?)
- num_arch_ok_M_Nm: number of archieved invoices with status "ok" between M and N months ago
- num_arch_rem_M_Nm: number of archieved invoices with status "rem" between M and N months ago (?)
- num_arch_written_off_M_Nm: number of archieved invoices that were unpaid between M and N months ago

**5. Order variables**
- merchant_category: category of the merchant receiving the order
- merchant_group: group of the merchant receiving the order