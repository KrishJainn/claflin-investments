"""5-Player Coach Dashboard Theme — light theme for better visibility."""

COACH_COLORS = {
    "background": "#ffffff",
    "card_bg": "#f8f9fa",
    "text": "#1a1a2e",
    "green": "#00a651",
    "red": "#dc3545",
    "blue": "#0066cc",
    "yellow": "#f0ad4e",
    "purple": "#6f42c1",
    "muted": "#6c757d",
    "border": "#dee2e6",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COACH_COLORS["background"],
    plot_bgcolor=COACH_COLORS["card_bg"],
    font=dict(color=COACH_COLORS["text"], family="Arial, sans-serif", size=12),
    xaxis=dict(gridcolor="#e9ecef", zeroline=False),
    yaxis=dict(gridcolor="#e9ecef", zeroline=False),
    margin=dict(l=50, r=20, t=40, b=40),
    hovermode="x unified",
)


def apply_theme(fig):
    """Apply Coach light theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig
