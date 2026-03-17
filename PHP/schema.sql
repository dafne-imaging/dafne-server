-- Dafne server – MySQL schema
-- Run once to initialise the database:
--   mysql -u root -p dafne < schema.sql

CREATE TABLE IF NOT EXISTS users (
    id           INT          NOT NULL AUTO_INCREMENT,
    name         VARCHAR(255) NOT NULL,
    email        VARCHAR(255) NOT NULL,
    -- Unsalted SHA-256 hex digest of the 16-character random API key.
    api_key_hash CHAR(64)     NOT NULL,
    -- Administration permission: allows changing permissions for other users.
    is_admin     TINYINT(1)   NOT NULL DEFAULT 0,
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_users_email        (email),
    UNIQUE KEY uq_users_api_key_hash (api_key_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Per-model access permissions (read canonical models, upload trained models).
-- A user may only interact with model types listed here.
CREATE TABLE IF NOT EXISTS users_accesspermissions (
    user_id    INT          NOT NULL,
    model_type VARCHAR(100) NOT NULL,

    PRIMARY KEY (user_id, model_type),
    CONSTRAINT fk_access_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Per-model merge permissions (list/download uploads, push merged models).
-- A user may only perform merge-client operations on model types listed here.
CREATE TABLE IF NOT EXISTS users_mergepermissions (
    user_id    INT          NOT NULL,
    model_type VARCHAR(100) NOT NULL,

    PRIMARY KEY (user_id, model_type),
    CONSTRAINT fk_merge_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;