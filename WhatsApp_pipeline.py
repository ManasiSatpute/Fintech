import re
import json
import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict


# CONFIGURATION
CHAT_FILE = "chat_2.txt"
DB_FILE = "bond_deals_2.db"

# For the current prototype, deterministic extraction is used.
# Later this can be replaced/combined with the LLM extraction layer.
USE_LLM = False


# DATA STRUCTURES

@dataclass
class Message:
    message_id: str
    timestamp: datetime
    sender: str
    text: str


@dataclass
class RequirementState:
    requirement_id: str

    bond: Optional[str] = None

    # BUY / SELL
    side: Optional[str] = None

    # Original requested quantity
    requested_amount: Optional[float] = None

    # Actual executed quantity
    executed_amount: Optional[float] = None

    # Final negotiated price/yield
    final_price: Optional[float] = None

    buyer: Optional[str] = None
    seller: Optional[str] = None

    status: str = "NEW"

    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    completion_time: Optional[datetime] = None

    # Keep all relevant messages for traceability
    message_ids: List[str] = field(default_factory=list)

    # Store important evidence
    evidence: Dict[str, List[str]] = field(default_factory=dict)


# GLOBAL MEMORY

# Requirement-level memory.
# This avoids sending the entire WhatsApp history to the LLM.
TRADE_MEMORY: Dict[str, RequirementState] = {}


# 1. WHATSAPP MESSAGE PARSER

def parse_whatsapp_chat(file_path: str) -> List[Message]:

    messages = []

    # Supports:
    # [11/08/26, 3:01:43 PM] Manasi Satpute: message
    #
    # Also works with Unicode spaces used in WhatsApp exports.

    pattern = re.compile(
        r"^\[(\d{1,2}/\d{1,2}/\d{2}),\s*"
        r"(\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M)\]\s*"
        r"([^:]+):\s*(.*)$"
    )

    current_message = None

    with open(file_path, "r", encoding="utf-8-sig") as f:

        for line in f:

            line = line.rstrip("\n")

            match = pattern.match(line)

            if match:

                date_str = match.group(1)
                time_str = match.group(2)
                sender = match.group(3).strip()
                text = match.group(4).strip()

                # Normalize unusual spaces
                time_str = (
                    time_str
                    .replace("\u202f", " ")
                    .replace("\xa0", " ")
                )

                try:
                    timestamp = datetime.strptime(
                        f"{date_str} {time_str}",
                        "%m/%d/%y %I:%M:%S %p"
                    )

                except ValueError:

                    timestamp = datetime.strptime(
                        f"{date_str} {time_str}",
                        "%m/%d/%y %I:%M %p"
                    )

                message_id = hashlib.sha1(
                    f"{timestamp}|{sender}|{text}".encode()
                ).hexdigest()[:12]

                current_message = Message(
                    message_id=message_id,
                    timestamp=timestamp,
                    sender=sender,
                    text=text
                )

                messages.append(current_message)

            else:

                # Handle multi-line WhatsApp messages
                if current_message and line.strip():
                    current_message.text += " " + line.strip()

    return messages


# ============================================================
# 2. REMOVE SYSTEM / NOISE MESSAGES
# ============================================================

SYSTEM_PATTERNS = [
    "Messages and calls are end-to-end encrypted",
    "You created group",
    "changed the subject",
    "changed the group",
    "added",
    "removed",
]


def is_noise_message(message: Message) -> bool:

    text = message.text.lower()

    for pattern in SYSTEM_PATTERNS:

        if pattern.lower() in text:
            return True

    return False


def clean_messages(messages: List[Message]) -> List[Message]:

    return [
        message
        for message in messages
        if not is_noise_message(message)
    ]


# ============================================================
# 3. BOND DETECTION
# ============================================================

BOND_PATTERN = re.compile(
    r"\b("
    r"\d+(?:\.\d+)?\s*"
    r"(?:GS|GSEC|SDL|T-BILL|TBILL|BOND)"
    r"(?:\s+\d{4})?"
    r")\b",
    re.IGNORECASE
)


def contains_bond_reference(text: str) -> bool:

    text_upper = text.upper()

    bond_keywords = [
        "GS",
        "GSEC",
        "BOND",
        "SDL",
        "T-BILL",
        "TBILL"
    ]

    return any(keyword in text_upper for keyword in bond_keywords)


def extract_bond(text: str) -> Optional[str]:

    match = BOND_PATTERN.search(text)

    if not match:
        return None

    bond = match.group(1)

    bond = re.sub(r"\s+", " ", bond).strip()

    return bond


# ============================================================
# 4. FINANCIAL RELEVANCE DETECTION
# ============================================================

TRADE_KEYWORDS = [
    "buy",
    "sell",
    "looking for",
    "looking",
    "bond",
    "gs",
    "gsec",
    "cr",
    "crore",
    "price",
    "done",
    "deal",
    "works",
    "offer",
    "yield"
]


def is_trade_relevant(text: str) -> bool:

    lower = text.lower()

    # Bond reference immediately makes message relevant
    if contains_bond_reference(text):
        return True

    # Explicit financial indicators
    for keyword in TRADE_KEYWORDS:

        if keyword in lower:
            return True

    # Price-like value
    if re.search(r"\b\d+\.\d{2,3}\b", text):
        return True

    # Amount like 10Cr / 5cr
    if re.search(
        r"\b\d+(?:\.\d+)?\s*(?:cr|crore|crores)\b",
        lower
    ):
        return True

    return False


# ============================================================
# 5. QUANTITY EXTRACTION
# ============================================================

QUANTITY_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(Cr|Crore|Crores)\b",
    re.IGNORECASE
)


def extract_quantity(text: str) -> Optional[float]:

    match = QUANTITY_PATTERN.search(text)

    if not match:
        return None

    return float(match.group(1))


# ============================================================
# 6. PRICE EXTRACTION
# ============================================================

PRICE_PATTERN = re.compile(
    r"\b\d+\.\d{2,3}\b"
)


def extract_prices(text: str) -> List[float]:

    values = []

    for match in PRICE_PATTERN.findall(text):

        try:

            value = float(match)

            values.append(value)

        except ValueError:
            pass

    return values


# ============================================================
# 7. IMPORTANT:
#    DISTINGUISH BOND COUPON FROM TRANSACTION PRICE
# ============================================================

def extract_coupon_from_bond(bond: Optional[str]) -> Optional[float]:

    if not bond:
        return None

    match = re.search(
        r"(\d+(?:\.\d+)?)",
        bond
    )

    if not match:
        return None

    return float(match.group(1))


def extract_transaction_price(
    text: str,
    bond: Optional[str]
) -> Optional[float]:

    prices = extract_prices(text)

    if not prices:
        return None

    coupon = extract_coupon_from_bond(bond)

    valid_prices = []

    for price in prices:

        # DO NOT use the coupon as transaction price
        if coupon is not None and abs(price - coupon) < 0.0001:
            continue

        # Year-like numbers are not relevant here
        if 1900 <= price <= 2100:
            continue

        valid_prices.append(price)

    if not valid_prices:
        return None

    return valid_prices[-1]


# ============================================================
# 8. SIDE DETECTION
# ============================================================

def extract_side(text: str) -> Optional[str]:

    lower = text.lower()

    if re.search(r"\bbuy\b", lower):
        return "BUY"

    if re.search(r"\bsell\b", lower):
        return "SELL"

    return None


# ============================================================
# 9. COMPLETION DETECTION
# ============================================================

COMPLETION_WORDS = [
    "done",
    "deal done",
    "confirmed",
    "confirm",
    "booked",
    "executed"
]


def is_completion_message(text: str) -> bool:

    lower = text.lower().strip()

    for word in COMPLETION_WORDS:

        if word in lower:
            return True

    return False


# ============================================================
# 10. REQUIREMENT ID
# ============================================================

def generate_requirement_id(
    bond: str,
    timestamp: datetime
) -> str:

    raw = f"{bond}|{timestamp.isoformat()}"

    digest = hashlib.sha256(
        raw.encode()
    ).hexdigest()[:10]

    return f"REQ-{digest.upper()}"


# ============================================================
# 11. TRANSACTION ID
# ============================================================

def generate_transaction_id(
    requirement_id: str,
    completion_time: datetime
) -> str:

    raw = (
        f"{requirement_id}|"
        f"{completion_time.isoformat()}"
    )

    digest = hashlib.sha256(
        raw.encode()
    ).hexdigest()[:12]

    return f"TXN-{digest.upper()}"


# ============================================================
# 12. EVIDENCE HELPER
# ============================================================

def add_evidence(
    state: RequirementState,
    field_name: str,
    message_id: str
):

    if field_name not in state.evidence:
        state.evidence[field_name] = []

    if message_id not in state.evidence[field_name]:

        state.evidence[field_name].append(message_id)


# ============================================================
# 13. FIND ACTIVE REQUIREMENT
# ============================================================

def find_active_requirement(
    bond: Optional[str],
    messages: List[Message],
    current_index: int
) -> Optional[RequirementState]:

    if not bond:
        return None

    # Most recent active requirement for same bond
    candidates = []

    for state in TRADE_MEMORY.values():

        if (
            state.bond
            and state.bond.lower() == bond.lower()
            and state.status not in ["COMPLETED", "CANCELLED"]
        ):
            candidates.append(state)

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x.last_updated or datetime.min,
        reverse=True
    )

    return candidates[0]


# ============================================================
# 14. CREATE REQUIREMENT
# ============================================================

def create_requirement(
    message: Message,
    bond: str
) -> RequirementState:

    requirement_id = generate_requirement_id(
        bond,
        message.timestamp
    )

    state = RequirementState(
        requirement_id=requirement_id,
        bond=bond,
        created_at=message.timestamp,
        last_updated=message.timestamp
    )

    TRADE_MEMORY[requirement_id] = state

    return state


# ============================================================
# 15. UPDATE REQUIREMENT MEMORY
# ============================================================

def update_requirement(
    state: RequirementState,
    message: Message
):

    text = message.text

    state.last_updated = message.timestamp

    if message.message_id not in state.message_ids:
        state.message_ids.append(message.message_id)

    # --------------------------------------------------------
    # SIDE
    # --------------------------------------------------------

    side = extract_side(text)

    if side:

        state.side = side

        add_evidence(
            state,
            "side",
            message.message_id
        )

    # --------------------------------------------------------
    # QUANTITY
    # --------------------------------------------------------

    quantity = extract_quantity(text)

    if quantity is not None:

        lower = text.lower()

        # If the requirement has not yet been created,
        # this is the requested quantity.
        if state.requested_amount is None:

            state.requested_amount = quantity

            add_evidence(
                state,
                "requested_amount",
                message.message_id
            )

        else:

            # Once a requirement exists, a quantity appearing
            # near completion represents executed quantity.
            if (
                is_completion_message(text)
                or "done" in lower
                or "executed" in lower
            ):

                state.executed_amount = quantity

                add_evidence(
                    state,
                    "executed_amount",
                    message.message_id
                )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    price = extract_transaction_price(
        text,
        state.bond
    )

    if price is not None:

        # Every later valid negotiated price replaces
        # the previous price.
        state.final_price = price

        add_evidence(
            state,
            "final_price",
            message.message_id
        )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    if is_completion_message(text):

        state.completion_time = message.timestamp

        # If quantity is less than requested,
        # transaction is partially filled.
        if (
            state.requested_amount is not None
            and state.executed_amount is not None
            and state.executed_amount < state.requested_amount
        ):

            state.status = "PARTIALLY_FILLED"

        else:

            state.status = "COMPLETED"

        add_evidence(
            state,
            "status",
            message.message_id
        )

    elif state.status == "NEW":

        state.status = "NEGOTIATING"


# ============================================================
# 16. PARTICIPANT IDENTIFICATION
# ============================================================

def determine_participants(
    state: RequirementState,
    messages: List[Message]
):

    # For this prototype:
    #
    # BUY:
    #   sender who says BUY -> buyer
    #   other participant -> seller
    #
    # SELL:
    #   sender who says SELL -> seller
    #   other participant -> buyer

    buy_sender = None
    sell_sender = None

    participants = set()

    for message in messages:

        participants.add(message.sender)

        side = extract_side(message.text)

        if side == "BUY":
            buy_sender = message.sender

        elif side == "SELL":
            sell_sender = message.sender

    if state.side == "BUY":

        if buy_sender:
            state.buyer = buy_sender

        # Find other participant
        for person in participants:

            if person != state.buyer:
                state.seller = person
                break

    elif state.side == "SELL":

        if sell_sender:
            state.seller = sell_sender

        for person in participants:

            if person != state.seller:
                state.buyer = person
                break


# ============================================================
# 17. VALIDATION
# ============================================================

def validate_transaction(
    state: RequirementState
) -> tuple:

    errors = []

    if not state.bond:
        errors.append("Bond is missing")

    if not state.side:
        errors.append("BUY/SELL side is missing")

    if not state.final_price:
        errors.append("Final price/yield is missing")

    if state.final_price is not None:

        if state.final_price <= 0:
            errors.append("Price must be positive")

    if not state.requested_amount:
        errors.append("Requested quantity is missing")

    if not state.executed_amount:
        errors.append("Executed quantity is missing")

    if not state.buyer:
        errors.append("Buyer is missing")

    if not state.seller:
        errors.append("Seller is missing")

    if state.buyer and state.seller:

        if state.buyer == state.seller:
            errors.append(
                "Buyer and seller cannot be same"
            )

    if state.status not in [
        "COMPLETED",
        "PARTIALLY_FILLED"
    ]:
        errors.append(
            "Transaction is not finalized"
        )

    return len(errors) == 0, errors


# ============================================================
# 18. DATABASE
# ============================================================

def create_database():

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    # --------------------------------------------------------
    # TRANSACTIONS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bond_deals_2 (

            transaction_id TEXT PRIMARY KEY,

            requirement_id TEXT,

            bond TEXT,

            coupon REAL,

            side TEXT,

            buyer TEXT,

            seller TEXT,

            requested_amount REAL,

            executed_amount REAL,

            final_price REAL,

            status TEXT,

            completion_time TEXT,

            created_at TEXT,

            evidence_json TEXT,

            message_ids_json TEXT,

            validation_status TEXT,

            validation_errors TEXT
        )
    """)

    # --------------------------------------------------------
    # REQUIREMENTS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requirements (

            requirement_id TEXT PRIMARY KEY,

            bond TEXT,

            side TEXT,

            requested_amount REAL,

            current_price REAL,

            buyer TEXT,

            seller TEXT,

            status TEXT,

            created_at TEXT,

            last_updated TEXT
        )
    """)

    conn.commit()

    return conn


# 19. SAVE REQUIREMENT


def save_requirement(
    conn,
    state: RequirementState
):

    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO requirements (
            requirement_id,
            bond,
            side,
            requested_amount,
            current_price,
            buyer,
            seller,
            status,
            created_at,
            last_updated
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,?)
    """, (

        state.requirement_id,

        state.bond,

        state.side,

        state.requested_amount,

        state.final_price,

        state.buyer,

        state.seller,

        state.status,

        state.created_at.isoformat()
        if state.created_at else None,

        state.last_updated.isoformat()
        if state.last_updated else None
    ))

    conn.commit()


# ============================================================
# 20. SAVE VALIDATED TRANSACTION
# ============================================================

def save_transaction(
    conn,
    state: RequirementState
):

    if not state.completion_time:
        return

    transaction_id = generate_transaction_id(
        state.requirement_id,
        state.completion_time
    )

    coupon = extract_coupon_from_bond(
        state.bond
    )

    valid, errors = validate_transaction(
        state
    )

    validation_status = (
        "VALID"
        if valid
        else "INVALID"
    )

    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO bond_deals_2 (

            transaction_id,
            requirement_id,
            bond,
            coupon,
            side,
            buyer,
            seller,
            requested_amount,
            executed_amount,
            final_price,
            status,
            completion_time,
            created_at,
            evidence_json,
            message_ids_json,
            validation_status,
            validation_errors

        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        transaction_id,

        state.requirement_id,

        state.bond,

        coupon,

        state.side,

        state.buyer,

        state.seller,

        state.requested_amount,

        state.executed_amount,

        state.final_price,

        state.status,

        state.completion_time.isoformat(),

        state.created_at.isoformat()
        if state.created_at else None,

        json.dumps(state.evidence),

        json.dumps(state.message_ids),

        validation_status,

        json.dumps(errors)
    ))

    conn.commit()

    return transaction_id


# ============================================================
# 21. PROCESS CHAT
# ============================================================

def process_chat(messages: List[Message]):

    conn = create_database()

    current_requirement = None

    for index, message in enumerate(messages):

        text = message.text

        if not is_trade_relevant(text):
            continue

        bond = extract_bond(text)

        # ----------------------------------------------------
        # If bond is found -> create/find requirement
        # ----------------------------------------------------

        if bond:

            current_requirement = find_active_requirement(
                bond,
                messages,
                index
            )

            if current_requirement is None:

                current_requirement = create_requirement(
                    message,
                    bond
                )

            # Evidence for bond
            add_evidence(
                current_requirement,
                "bond",
                message.message_id
            )

        # ----------------------------------------------------
        # If no bond on current message, use current context
        # ----------------------------------------------------

        if current_requirement is None:

            continue

        # ----------------------------------------------------
        # Update memory using current message
        # ----------------------------------------------------

        update_requirement(
            current_requirement,
            message
        )

        # ----------------------------------------------------
        # Save requirement state
        # ----------------------------------------------------

        determine_participants(
            current_requirement,
            messages
        )

        save_requirement(
            conn,
            current_requirement
        )

        # ----------------------------------------------------
        # If finalized -> validate and save transaction
        # ----------------------------------------------------

        if current_requirement.status in [
            "COMPLETED",
            "PARTIALLY_FILLED"
        ]:

            valid, errors = validate_transaction(
                current_requirement
            )

            if valid:

                transaction_id = save_transaction(
                    conn,
                    current_requirement
                )

                print("\n" + "=" * 70)
                print("VALIDATED TRANSACTION")
                print("=" * 70)

                print(
                    f"Transaction ID    : {transaction_id}"
                )

                print(
                    f"Requirement ID    : "
                    f"{current_requirement.requirement_id}"
                )

                print(
                    f"Bond              : "
                    f"{current_requirement.bond}"
                )

                print(
                    f"Coupon            : "
                    f"{extract_coupon_from_bond(current_requirement.bond)}"
                )

                print(
                    f"Side              : "
                    f"{current_requirement.side}"
                )

                print(
                    f"Buyer             : "
                    f"{current_requirement.buyer}"
                )

                print(
                    f"Seller            : "
                    f"{current_requirement.seller}"
                )

                print(
                    f"Requested Amount  : "
                    f"{current_requirement.requested_amount} Cr"
                )

                print(
                    f"Executed Amount   : "
                    f"{current_requirement.executed_amount} Cr"
                )

                print(
                    f"Final Price       : "
                    f"{current_requirement.final_price}"
                )

                print(
                    f"Status            : "
                    f"{current_requirement.status}"
                )

                print(
                    f"Completion Time   : "
                    f"{current_requirement.completion_time}"
                )

                print("\nEvidence:")

                for field_name, ids in (
                    current_requirement.evidence.items()
                ):

                    print(
                        f"  {field_name}: {ids}"
                    )

                print("=" * 70)

            else:

                print("\nTRANSACTION NOT SAVED")
                print("Validation errors:")

                for error in errors:
                    print(f" - {error}")

    conn.close()


# ============================================================
# 22. EOD REPORT
# ============================================================

def generate_eod_report():

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            status,
            COUNT(*),
            SUM(executed_amount)
        FROM bond_deals_2
        GROUP BY status
    """)

    rows = cursor.fetchall()

    print("\n")
    print("=" * 70)
    print("EOD TRANSACTION REPORT")
    print("=" * 70)

    if not rows:

        print("No validated transactions found.")

    else:

        for status, count, quantity in rows:

            print(
                f"{status:20} "
                f"Transactions: {count:3} "
                f"Quantity: {quantity or 0} Cr"
            )

    print("=" * 70)

    conn.close()


# ============================================================
# 23. DISPLAY DATABASE
# ============================================================

def display_database():

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            transaction_id,
            bond,
            side,
            buyer,
            seller,
            requested_amount,
            executed_amount,
            final_price,
            status
        FROM bond_deals_2
    """)

    rows = cursor.fetchall()

    print("\n")
    print("=" * 100)
    print("DATABASE RECORDS")
    print("=" * 100)

    if not rows:

        print("No records found.")

    else:

        for row in rows:

            print(
                f"""
Transaction ID   : {row[0]}
Bond             : {row[1]}
Side             : {row[2]}
Buyer            : {row[3]}
Seller           : {row[4]}
Requested        : {row[5]} Cr
Executed         : {row[6]} Cr
Final Price      : {row[7]}
Status           : {row[8]}
"""
            )

    print("=" * 100)

    conn.close()


# ============================================================
# 24. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("WHATSAPP BOND TRANSACTION INTELLIGENCE")
    print("=" * 70)

    print("\nReading chat...")

    messages = parse_whatsapp_chat(
        CHAT_FILE
    )

    print(
        f"Total messages parsed: {len(messages)}"
    )

    messages = clean_messages(
        messages
    )

    print(
        f"Messages after noise removal: "
        f"{len(messages)}"
    )

    print("\nProcessing trade conversations...")

    process_chat(
        messages
    )

    generate_eod_report()

    display_database()


if __name__ == "__main__":
    main()