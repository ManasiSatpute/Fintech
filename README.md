# WhatsApp Bond Trade Intelligence System

An AI-powered fintech pipeline that extracts bond-related trade information from WhatsApp chat logs, validates the extracted financial entities, and stores structured trade records in a SQL database.

The system is designed to handle noisy WhatsApp conversations where trade information may be distributed across multiple messages, while maintaining context, reducing unnecessary LLM calls, and enforcing strict financial-data validation.

---

## Project Overview

Bond dealers and traders often communicate through WhatsApp using short, informal, and fragmented messages such as:

```text
ABC Bank
BUY 5Cr
8.25% GS 2034
YTM 7.12
```

or:

```text
Need 10cr
7.40% 2035
at 98.25
```

Manually converting these conversations into structured trade records is time-consuming and error-prone.

This project automates that process using an **Agentic AI pipeline**.

### Core Workflow

```text
WhatsApp Chat Logs
        │
        ▼
┌──────────────────────┐
│ Chat Preprocessing   │
│ & Noise Removal      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Deterministic Router │
│ Agent                │
└──────────┬───────────┘
           │
      Bond Related?
       /        \
     No          Yes
     │            │
   Ignore         ▼
          ┌──────────────────┐
          │ Context Manager  │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Extraction Agent │
          │ Structured JSON  │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Validation Agent │
          └────────┬─────────┘
                   │
             Valid / Invalid
                /       \
              Valid     Invalid
                │          │
                ▼          ▼
          ┌──────────┐   REJECT
          │ SQL DB   │
          └──────────┘
```

---

## Key Features

* Detects bond-related messages from WhatsApp conversations.
* Filters non-financial and irrelevant messages before LLM processing.
* Handles fragmented trade information across multiple messages.
* Extracts structured financial entities using typed schemas.
* Validates extracted values before database insertion.
* Supports BUY, SELL, AXE, INQUIRY and TWO-WAY intents.
* Calculates/stores extraction confidence.
* Maintains processing and validation status.
* Uses parameterized SQL queries.
* Maintains an audit trail for traceability.
* Supports PII masking before external LLM processing.
* Designed for low latency and reduced LLM token consumption.

---

## Financial Information Extracted

The extraction pipeline identifies the following entities:

| Field             | Description                                     |
| ----------------- | ----------------------------------------------- |
| Instrument Name   | Bond/instrument name                            |
| Issuer            | Bond issuing organization                       |
| ISIN              | International Securities Identification Number  |
| CUSIP             | CUSIP identifier, if available                  |
| Bond Type         | Sovereign, Corporate, Municipal, Eurobond, etc. |
| Action / Intent   | BUY, SELL, AXE, INQUIRY, TWO-WAY                |
| Quantity          | Bond quantity                                   |
| Face Value        | Face/notional value                             |
| Price             | Clean/dirty price                               |
| Yield             | Bond yield/YTM                                  |
| Spread            | Spread, if available                            |
| Coupon Rate       | Coupon percentage                               |
| Maturity Date     | Bond maturity date                              |
| Counterparty      | Sender/counterparty information                 |
| Confidence Score  | Extraction confidence from 0.0–1.0              |
| Validation Status | Result of financial validation                  |
| Processing Status | PENDING, VERIFIED or REJECTED                   |

---

## Agentic Architecture

### 1. Preprocessing Layer

Raw WhatsApp exports contain considerable noise:

* Timestamps
* Sender metadata
* System messages
* Emojis
* Media notifications
* Repeated messages
* Unnecessary whitespace

The preprocessing layer cleans and compresses the conversation before sending relevant content to AI models.

```text
Raw WhatsApp Export
        │
        ├── Remove system messages
        ├── Normalize whitespace
        ├── Remove unnecessary metadata
        ├── Normalize financial notation
        └── Preserve sender/message ordering
                │
                ▼
          Cleaned Messages
```

### 2. Router Agent

The Router Agent is intentionally lightweight and deterministic where possible.

Its purpose is to answer:

```text
Is this conversation/message potentially related
to a bond trade or bond inquiry?
```

Irrelevant conversations are discarded before expensive extraction operations.

This reduces:

* LLM calls
* Token consumption
* Processing latency
* Infrastructure cost

### 3. Context Manager

Bond information can be distributed across multiple WhatsApp messages.

Example:

```text
Trader A: 7.40 GS 2035
Trader B: 98.25
Trader A: 5 Cr
Trader B: BUY?
```

Processing each message independently could result in incomplete extraction.

The context manager therefore groups related messages using factors such as:

* Sender
* Timestamp
* Message proximity
* Conversation/session
* Instrument references
* Trade intent

The grouped context is then provided to the extraction agent.

### 4. Extraction Agent

The Extraction Agent converts unstructured text into structured financial information.

Example output:

```json
{
  "instrument": "7.40% GS 2035",
  "issuer": "Government of India",
  "isin": null,
  "bond_type": "Sovereign",
  "action": "BUY",
  "quantity": "5 Cr",
  "price": 98.25,
  "yield": null,
  "coupon_rate": 7.40,
  "maturity_date": "2035",
  "confidence_score": 0.94
}
```

Structured schemas are used to prevent malformed or conversational LLM output.

### 5. Validation Agent

The Validation Agent verifies the extracted information before persistence.

Validation can include:

* ISIN format validation
* CUSIP format validation
* Numeric field validation
* Coupon-rate validation
* Price validation
* Yield/price consistency checks
* Maturity-date validation
* Required-field validation
* Action/intent validation
* Confidence threshold validation

Only validated records are persisted as verified trades.

---

## Database Design

The system uses a relational SQL database to maintain ACID-compliant financial records.

A typical trade record contains:

```text
Trade
 ├── Trade ID
 ├── Instrument
 ├── Issuer
 ├── ISIN
 ├── CUSIP
 ├── Bond Type
 ├── Action
 ├── Quantity
 ├── Face Value
 ├── Price
 ├── Yield
 ├── Coupon
 ├── Maturity
 ├── Counterparty
 ├── Confidence Score
 ├── Validation Status
 ├── Processing Status
 └── Created At
```

### Processing Status

```text
PENDING
   │
   ▼
VALIDATING
   │
   ├── Valid ──────► VERIFIED
   │
   └── Invalid ────► REJECTED
```

Database constraints, foreign keys, unique indexes and parameterized queries should be used to protect data integrity.

---

## Security

Because the system processes financial conversations, security is a core component.

### PII Masking

Sensitive information such as:

```text
Phone numbers
Personal names
User identifiers
```

should be anonymized or hashed before sending data to external LLM APIs.

Example:

```text
Original:
Rahul: BUY 5Cr from +91XXXXXXXXXX

Sanitized:
TRADER_01: BUY 5Cr from COUNTERPARTY_01
```

### SQL Injection Protection

All database operations should use parameterized queries rather than dynamically constructed SQL.

```python
cursor.execute(
    """
    INSERT INTO trades (isin, action, quantity)
    VALUES (?, ?, ?)
    """,
    (isin, action, quantity)
)
```

### Secrets Management

Credentials should never be hardcoded.

Use:

```text
Environment Variables
        or
Secret Manager / Key Vault
```

### Audit Trail

The system should maintain an immutable audit trail containing:

```text
Raw Chat Payload
       │
       ▼
Sanitized Input
       │
       ▼
LLM Extraction Output
       │
       ▼
Validation Result
       │
       ▼
Database Commit Status
```

This provides traceability for every processed trade.

---

## Technology Stack

| Component              | Technology                        |
| ---------------------- | --------------------------------- |
| Language               | Python                            |
| AI/LLM                 | OpenAI / Lightweight LLM          |
| Agent Framework        | LangGraph / Custom Agent Pipeline |
| Validation             | Pydantic                          |
| Database               | MySQL / PostgreSQL                |
| Data Processing        | Python                            |
| API Layer              | FastAPI                           |
| Environment Management | `.env` / Secret Manager           |
| Version Control        | Git / GitHub                      |

The exact LLM and database implementation can be configured according to deployment requirements.

---

## Project Structure

```text
WhatsApp-to-SQL/
│
├── WhatsApp_pipeline.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── agents/
│   ├── router_agent.py
│   ├── extraction_agent.py
│   └── validation_agent.py
│
├── preprocessing/
│   └── whatsapp_cleaner.py
│
├── database/
│   ├── connection.py
│   ├── models.py
│   └── repository.py
│
├── schemas/
│   └── trade_schema.py
│
├── utils/
│   ├── logger.py
│   ├── pii_masking.py
│   └── validators.py
│
└── tests/
    ├── test_router.py
    ├── test_extraction.py
    └── test_validation.py
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd WhatsApp-to-SQL
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
DB_HOST=localhost
DB_PORT=3306
DB_NAME=bond_trades
DB_USER=your_username
DB_PASSWORD=your_password
```

Never commit `.env` to GitHub.

---

## Running the Pipeline

Run the main pipeline:

```bash
python3 WhatsApp_pipeline.py
```

The pipeline performs:

```text
Input Chat
   ↓
Preprocessing
   ↓
Bond Detection
   ↓
Context Grouping
   ↓
Entity Extraction
   ↓
Validation
   ↓
SQL Persistence
   ↓
Audit Logging
```

---

## Example

### Input

```text
10:21 AM Trader A:
7.40% GS 2035

10:22 AM Trader B:
98.25

10:23 AM Trader A:
BUY 5 Cr
```

### Extracted Record

```json
{
  "instrument": "7.40% GS 2035",
  "bond_type": "Sovereign",
  "action": "BUY",
  "quantity": "5 Cr",
  "price": 98.25,
  "coupon_rate": 7.40,
  "confidence_score": 0.95,
  "validation_status": "VALID",
  "processing_status": "VERIFIED"
}
```

---

## Error Handling

The pipeline should handle failures without compromising financial data integrity.

Examples:

```text
Invalid JSON
     ↓
Retry / Reject

Invalid ISIN
     ↓
Validation Failure

LLM Timeout
     ↓
Retry with Backoff

Database Failure
     ↓
Rollback Transaction

Low Confidence
     ↓
PENDING / Manual Review
```

Database transactions should use commit/rollback semantics to prevent partial records.

---

## Performance & Token Optimization

The system follows a multi-stage filtering architecture:

```text
All WhatsApp Messages
        │
        ▼
Deterministic Filtering
        │
        ▼
Potential Bond Messages
        │
        ▼
Lightweight Router
        │
        ▼
Relevant Conversations
        │
        ▼
LLM Extraction
        │
        ▼
Validation
```

This prevents every WhatsApp message from being sent to a large language model.

Additional optimization techniques include:

* Removing redundant metadata
* Compressing conversation context
* Using lightweight models for routing
* Strict structured-output schemas
* Avoiding unnecessary few-shot examples
* Processing only relevant conversation windows

---

## Reliability Considerations

For production fintech deployment, the system should support:

* Idempotent processing
* Transaction-safe database writes
* Duplicate detection
* Retry mechanisms
* Dead-letter/error queues
* Structured logging
* Monitoring and alerting
* Model/version tracking
* Auditability
* Manual review for low-confidence trades

---

## Future Enhancements

Potential improvements include:

* Real-time WhatsApp ingestion
* Kafka/RabbitMQ-based event processing
* Human-in-the-loop trade verification
* Redis-based context/session management
* FastAPI production API
* LangGraph-based stateful agent orchestration
* Model fallback mechanisms
* Advanced duplicate-trade detection
* Trade analytics dashboard
* Role-based access control
* LLM gateway for centralized model governance
* Model evaluation and monitoring
* Automated regression testing for extraction accuracy

---

## Disclaimer

This project is intended for **educational, research, and prototype purposes**.

Financial trade extraction systems deployed in production should undergo appropriate security, compliance, validation, model-risk, data-governance, and operational reviews before being used with real financial transactions.
---

