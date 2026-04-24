import streamlit as st
from dataclasses import dataclass, field

NODE_OUTPUT_LABELS = {
    "planning": "📋 Plan",
    "generate_joke": "😄 Joke",
}


@dataclass
class ChatMessage:
    """Represents a single chat message, optionally with node outputs."""
    role: str
    content: str
    node_outputs: dict[str, str] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    planning_required: bool | None = None
    trace_steps: list[dict] = field(default_factory=list)

    def render(self):
        """Renders the chat message in Streamlit, including any node outputs in expanders."""
        with st.chat_message(self.role):
            if self.trace_steps:
                with st.expander(f"Agent trace ({len(self.trace_steps)} steps)", expanded=False):
                    for step in self.trace_steps:
                        duration = f"  `{step['duration_ms']} ms`" if "duration_ms" in step else ""
                        st.markdown(f"✅ {step['label']}{duration}")

            st.markdown(self.content)

            if self.planning_required is not None:
                label, color = ("🗺 Planned", "green") if self.planning_required else ("⚡ Direct", "orange")
                st.badge(label, color=color)


            if self.node_outputs:
                for node, content in self.node_outputs.items():
                    label = NODE_OUTPUT_LABELS.get(node, node)
                    with st.expander(label):
                        st.markdown(content)
            if self.stats:
                st.caption(
                    f"⏱ {self.stats['elapsed_seconds']}s · "
                    f"↑ {self.stats['input_tokens']} tokens in · "
                    f"↓ {self.stats['output_tokens']} tokens out"
                )
