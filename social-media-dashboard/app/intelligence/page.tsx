import { supabase } from "@/lib/supabase"

export const revalidate = 0

interface SelectionRow {
  hook_type:        string | null
  topic_category:   string | null
  performance_tier: string | null
  views:            number | null
}

interface RankItem {
  label: string
  avg:   number
  top:   number
  total: number
}

function buildRankings(rows: SelectionRow[], key: "hook_type" | "topic_category"): RankItem[] {
  const map: Record<string, { views: number[]; top: number; total: number }> = {}
  for (const r of rows) {
    const k = r[key] || "unknown"
    if (!map[k]) map[k] = { views: [], top: 0, total: 0 }
    if (r.performance_tier === "top") map[k].top++
    map[k].total++
    if (r.views) map[k].views.push(r.views)
  }
  return Object.entries(map)
    .map(([label, s]) => ({
      label,
      avg:   s.views.length ? Math.round(s.views.reduce((a, b) => a + b, 0) / s.views.length) : 0,
      top:   s.top,
      total: s.total,
    }))
    .sort((a, b) => b.avg - a.avg)
}

function fmt(n: number) {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)
}

function timeAgo(iso: string) {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function cleanAnalysis(raw: string): { heading: string; paragraphs: string[] }[] {
  const lines = raw.split("\n")
  const sections: { heading: string; paragraphs: string[] }[] = []
  let currentHeading = ""
  let currentLines: string[] = []

  function flush() {
    const text = currentLines
      .join(" ")
      .replace(/\s+/g, " ")
      .trim()
    if (text.length > 40) {
      sections.push({ heading: currentHeading, paragraphs: [text] })
    }
    currentLines = []
  }

  for (const raw of lines) {
    const line = raw
      .replace(/^#{1,6}\s*/g, "")          // remove # headers
      .replace(/\*\*(.*?)\*\*/g, "$1")      // remove bold
      .replace(/\*(.*?)\*/g, "$1")          // remove italic
      .replace(/^---+$/, "")               // remove dividers
      .replace(/^\|.*$/, "")               // remove table rows
      .replace(/^[-|:]+$/, "")             // remove table separators
      .replace(/^For:.*$/, "")             // remove "For: AI" lines
      .replace(/Read this entire.*$/, "")  // remove meta instructions
      .replace(/Every decision.*$/, "")    // remove meta instructions
      .trim()

    if (!line) {
      if (currentLines.length > 0) {
        flush()
      }
      continue
    }

    // Detect section headings: short ALL-CAPS lines or numbered headings
    const isHeading = line.length < 80 && (
      /^\d+\.\s/.test(line) ||
      (line === line.toUpperCase() && line.length > 4 && !/^\d+$/.test(line))
    )

    if (isHeading) {
      flush()
      currentHeading = line.replace(/^\d+\.\s*/, "").replace(/[—–-]+.*$/, "").trim()
    } else {
      // Skip lines that look like leftovers of tables or list bullets with just numbers
      if (/^\|/.test(line) || /^[-*•]\s*$/.test(line)) continue
      // Clean bullet points
      const clean = line.replace(/^[-*•]\s+/, "")
      if (clean.length > 10) currentLines.push(clean)
    }
  }
  flush()

  // Remove sections that are just the hook/topic ranking tables (already shown visually)
  return sections.filter(s =>
    !s.heading.toLowerCase().includes("ranking") &&
    !s.heading.toLowerCase().includes("cross-platform average") &&
    s.paragraphs.some(p => p.length > 60)
  )
}

const TIER_COLOR = { top: "#16a34a", mid: "#d97706", low: "#dc2626" }

function RankBars({ items }: { items: RankItem[] }) {
  const max = Math.max(...items.map(r => r.avg), 1)
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
      {items.map(({ label, avg, top, total }, i) => {
        const pct    = Math.round((avg / max) * 100)
        const topPct = total ? Math.round((top / total) * 100) : 0
        const trendColor = topPct >= 40 ? TIER_COLOR.top : topPct >= 20 ? TIER_COLOR.mid : "var(--faint)"
        return (
          <div key={label}>
            <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "5px" }}>
              <span style={{ width: "110px", fontSize: "12px", fontWeight: i === 0 ? 600 : 400, color: "var(--text)", textTransform: "capitalize", flexShrink: 0 }}>
                {label}
              </span>
              <div style={{ flex: 1, height: "2px", background: "var(--border)", borderRadius: "2px" }}>
                <div style={{
                  height: "100%",
                  width: `${pct}%`,
                  minWidth: avg > 0 ? "3px" : 0,
                  background: i === 0 ? "var(--text)" : "var(--faint)",
                  borderRadius: "2px",
                  transformOrigin: "left center",
                  animation: `barGrow 0.7s ease both`,
                  animationDelay: `${i * 0.08}s`,
                }} />
              </div>
              <span style={{ fontSize: "12px", color: "var(--muted)", fontVariantNumeric: "tabular-nums", width: "52px", textAlign: "right", flexShrink: 0 }}>
                {fmt(avg)} avg
              </span>
              <span style={{ fontSize: "11px", color: trendColor, width: "42px", textAlign: "right", flexShrink: 0 }}>
                {topPct}% top
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default async function IntelligencePage() {
  const [intelligenceRes, logRes] = await Promise.all([
    supabase.from("channel_intelligence").select("summary, stats, updated_at").eq("id", "singleton").single(),
    supabase.from("clip_selection_log")
      .select("hook_type, topic_category, performance_tier, views")
      .not("performance_tier", "is", null),
  ])

  const intel  = intelligenceRes.data
  const stats  = (intel?.stats ?? {}) as Record<string, number | string>
  const rows   = (logRes.data ?? []) as SelectionRow[]

  const hookRankings  = buildRankings(rows, "hook_type")
  const topicRankings = buildRankings(rows, "topic_category")

  const tierCounts = rows.reduce((acc, r) => {
    if (r.performance_tier) acc[r.performance_tier] = (acc[r.performance_tier] || 0) + 1
    return acc
  }, {} as Record<string, number>)
  const total = rows.length

  const analysisSections = intel?.summary ? cleanAnalysis(intel.summary) : []

  return (
    <div style={{ maxWidth: "720px" }}>

      {/* Header */}
      <div style={{ marginBottom: "4rem" }}>
        <p style={{ fontSize: "11px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--faint)", marginBottom: "1.5rem" }}>
          Intelligence
          {intel?.updated_at && ` · Updated ${timeAgo(intel.updated_at)}`}
        </p>
        <h1 style={{ fontSize: "clamp(2rem, 4vw, 2.75rem)", fontWeight: 300, lineHeight: 1.12, letterSpacing: "-0.035em", color: "var(--text)", marginBottom: "1.25rem" }}>
          What the agent<br />has learned.
        </h1>
        <p style={{ fontSize: "1rem", color: "var(--muted)", lineHeight: 1.7, maxWidth: "480px" }}>
          {total} clips scored · {fmt(Number(stats.total_views || 0))} total views · {Number(stats.avg_engagement || 0).toFixed(1)}% avg engagement
        </p>
      </div>

      {/* Performance split */}
      <section style={{ marginBottom: "4rem" }}>
        <p style={{ fontSize: "11px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--faint)", marginBottom: "1.75rem" }}>
          Performance Split
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1px", background: "var(--border)", border: "1px solid var(--border)", marginBottom: "1rem" }}>
          {(["top", "mid", "low"] as const).map(tier => {
            const count = tierCounts[tier] || 0
            const pct   = total ? Math.round((count / total) * 100) : 0
            return (
              <div key={tier} style={{ background: "var(--bg)", padding: "1.25rem 1.5rem" }}>
                <div style={{ fontSize: "11px", color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.5rem" }}>
                  {tier === "top" ? "Top" : tier === "mid" ? "Mid" : "Low"}
                </div>
                <div style={{ fontSize: "1.8rem", fontWeight: 300, letterSpacing: "-0.04em", color: TIER_COLOR[tier], marginBottom: "2px" }}>
                  {pct}%
                </div>
                <div style={{ fontSize: "11px", color: "var(--faint)" }}>{count} clips</div>
              </div>
            )
          })}
        </div>
        <div style={{ display: "flex", height: "2px", overflow: "hidden" }}>
          {(["top", "mid", "low"] as const).map(tier => (
            <div key={tier} style={{ width: `${total ? (tierCounts[tier] || 0) / total * 100 : 0}%`, background: TIER_COLOR[tier] }} />
          ))}
        </div>
      </section>

      {/* Hook rankings */}
      <section style={{ marginBottom: "4rem" }}>
        <p style={{ fontSize: "11px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--faint)", marginBottom: "1.75rem" }}>
          Hook Type Rankings
        </p>
        <RankBars items={hookRankings} />
      </section>

      {/* Topic rankings */}
      <section style={{ marginBottom: "4rem" }}>
        <p style={{ fontSize: "11px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--faint)", marginBottom: "1.75rem" }}>
          Topic Rankings
        </p>
        <RankBars items={topicRankings} />
      </section>

      {/* Best hook */}
      {stats.best_hook && (
        <section style={{ marginBottom: "4rem" }}>
          <p style={{ fontSize: "11px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--faint)", marginBottom: "1.25rem" }}>
            Best Performing Hook · {fmt(Number(stats.best_views))} views
          </p>
          <div style={{ borderLeft: "2px solid var(--text)", paddingLeft: "1.25rem" }}>
            <p style={{ fontSize: "1.1rem", fontWeight: 300, color: "var(--text)", lineHeight: 1.65, margin: 0, letterSpacing: "-0.01em" }}>
              &ldquo;{String(stats.best_hook)}&rdquo;
            </p>
          </div>
        </section>
      )}

      {/* For-Ted recommendations -- human review, never auto-applied */}
      {analysisSections.filter(s => s.heading.toLowerCase().includes("for ted")).map((s, i) => (
        <section key={`recs-${i}`} style={{
          marginBottom: "4rem",
          border: "1px solid #f59e0b",
          borderLeft: "3px solid #f59e0b",
          borderRadius: "2px",
          padding: "1.5rem 1.75rem",
          background: "rgba(245, 158, 11, 0.06)",
        }}>
          <p style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "0.14em", textTransform: "uppercase", color: "#f59e0b", marginBottom: "1rem" }}>
            For Ted &middot; Recommended Actions &middot; Not Auto-Applied
          </p>
          {s.paragraphs.map((p, j) => (
            <p key={j} style={{ fontSize: "14px", color: "var(--text)", lineHeight: 1.8, margin: 0, marginBottom: j < s.paragraphs.length - 1 ? "1rem" : 0 }}>
              {p}
            </p>
          ))}
        </section>
      ))}

      {/* Analysis */}
      {analysisSections.filter(s => !s.heading.toLowerCase().includes("for ted")).length > 0 && (
        <section>
          <p style={{ fontSize: "11px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--faint)", marginBottom: "2rem" }}>
            Channel Insights
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "0" }}>
            {analysisSections.filter(s => !s.heading.toLowerCase().includes("for ted")).map((s, i) => (
              <div key={i} style={{ borderTop: "1px solid var(--border)", paddingTop: "1.75rem", paddingBottom: "1.75rem" }}>
                {s.heading && (
                  <p style={{ fontSize: "12px", fontWeight: 600, color: "var(--text)", marginBottom: "0.6rem", textTransform: "capitalize" }}>
                    {s.heading.toLowerCase().replace(/\b\w/g, c => c.toUpperCase())}
                  </p>
                )}
                {s.paragraphs.map((p, j) => (
                  <p key={j} style={{ fontSize: "14px", color: "var(--muted)", lineHeight: 1.8, margin: 0, marginBottom: j < s.paragraphs.length - 1 ? "1rem" : 0 }}>
                    {p}
                  </p>
                ))}
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
