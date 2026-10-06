-- ============================================================================
-- E-commerce RFM Customer Segmentation
-- ----------------------------------------------------------------------------
-- Computes Recency / Frequency / Monetary per customer, scores each dimension
-- into quintiles with NTILE(), builds an RFM total score and assigns a
-- marketing segment, then reports segment-level revenue share.
--
-- Reference date: 2026-10-01 (day after the last order in the dataset).
-- Tested on SQLite 3.x and PostgreSQL 12+.
--
-- Note on NTILE: quintiles are row-based, so with a skewed customer base
-- (many one-time buyers) the same raw value (e.g. frequency = 1) can span
-- more than one quintile. This is standard behaviour and is called out in
-- the README.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Customer-level RFM base metrics
-- ---------------------------------------------------------------------------
WITH rfm_base AS (
    SELECT
        customer_id,
        CAST(julianday('2026-10-01') - julianday(MAX(order_date)) AS INTEGER)
            AS recency_days,                 -- days since last purchase
        COUNT(*)              AS frequency,  -- total orders
        ROUND(SUM(amount_inr), 2) AS monetary -- total spend (INR)
    FROM orders
    GROUP BY customer_id
),

-- ---------------------------------------------------------------------------
-- 2. Quintile scoring with NTILE().
--    Recency is inverted: the MOST recent buyers land in quintile 5.
-- ---------------------------------------------------------------------------
rfm_scored AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,  -- 5 = most recent
        NTILE(5) OVER (ORDER BY frequency)         AS f_score,  -- 5 = most orders
        NTILE(5) OVER (ORDER BY monetary)         AS m_score    -- 5 = biggest spend
    FROM rfm_base
),

-- ---------------------------------------------------------------------------
-- 3. Segment assignment.
--    Champions = elite RFM total (>= 14 of 15). Order matters: the most
--    valuable / most urgent segments are matched first.
-- ---------------------------------------------------------------------------
rfm_segmented AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary,
        r_score,
        f_score,
        m_score,
        CAST(r_score AS TEXT) || CAST(f_score AS TEXT) || CAST(m_score AS TEXT)
            AS rfm_score,
        (r_score + f_score + m_score) AS rfm_total,
        CASE
            WHEN (r_score + f_score + m_score) >= 14
                THEN 'Champions'
            WHEN r_score <= 2 AND f_score >= 4 AND m_score >= 4
                THEN 'Can''t Lose Them'
            WHEN r_score <= 2 AND f_score >= 3
                THEN 'At Risk'
            WHEN f_score >= 4
                THEN 'Loyal Customers'
            WHEN r_score >= 4 AND f_score = 3
                THEN 'Potential Loyalists'
            WHEN r_score = 5 AND f_score <= 2
                THEN 'New Customers'
            WHEN r_score = 4 AND f_score <= 2
                THEN 'Promising'
            WHEN r_score = 3 AND f_score >= 3
                THEN 'Need Attention'
            WHEN r_score = 2 AND f_score <= 2
                THEN 'About to Sleep'
            WHEN r_score = 1
                THEN 'Lost'
            ELSE 'Hibernating'
        END AS segment
    FROM rfm_scored
)

-- ---------------------------------------------------------------------------
-- 4. Segment-level summary: size, revenue and revenue share
-- ---------------------------------------------------------------------------
SELECT
    segment,
    COUNT(*) AS customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_customers,
    ROUND(SUM(monetary), 2)                            AS total_revenue_inr,
    ROUND(100.0 * SUM(monetary) / SUM(SUM(monetary)) OVER (), 1)
                                                      AS pct_revenue,
    ROUND(AVG(monetary), 2)                           AS avg_revenue_per_customer,
    ROUND(AVG(recency_days), 1)                        AS avg_recency_days,
    ROUND(AVG(frequency), 1)                          AS avg_frequency
FROM rfm_segmented
GROUP BY segment
ORDER BY total_revenue_inr DESC;
