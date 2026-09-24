"""
3D Holographic Brain & Neural Network Visual Component.
Pure SVG & CSS medical-AI visual with depth, animated scanning beam, and telemetry HUD.
Zero external network dependencies, lightweight, responsive.
"""


def get_3d_brain_svg() -> str:
    """Generate interactive 3D holographic brain HUD SVG."""
    return """
<div class="hologram-stage">
  <div class="hologram-glow"></div>
  <svg class="hologram-svg" viewBox="0 0 520 400" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <!-- Gradients -->
      <linearGradient id="cyberGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#00D4FF" stop-opacity="0.9" />
        <stop offset="50%" stop-color="#2563EB" stop-opacity="0.7" />
        <stop offset="100%" stop-color="#7C3AED" stop-opacity="0.8" />
      </linearGradient>
      <linearGradient id="scanBeamGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#00D4FF" stop-opacity="0" />
        <stop offset="50%" stop-color="#00D4FF" stop-opacity="0.8" />
        <stop offset="100%" stop-color="#00D4FF" stop-opacity="0" />
      </linearGradient>
      <radialGradient id="brainCenterGlow" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.35" />
        <stop offset="60%" stop-color="#0284C7" stop-opacity="0.12" />
        <stop offset="100%" stop-color="#0369A1" stop-opacity="0" />
      </radialGradient>
      <filter id="neonGlow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3.5" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>

    <!-- Outer 3D Orbital Projection Rings -->
    <ellipse cx="260" cy="200" rx="220" ry="90" fill="none" stroke="#0284C7" stroke-width="1.2" stroke-dasharray="6,4" opacity="0.45" />
    <ellipse cx="260" cy="200" rx="195" ry="80" fill="none" stroke="#38BDF8" stroke-width="1" stroke-dasharray="14,8" opacity="0.3" />
    
    <!-- Depth Base Platform -->
    <ellipse cx="260" cy="285" rx="170" ry="45" fill="none" stroke="#2563EB" stroke-width="1.5" stroke-dasharray="4,4" opacity="0.4" />
    <ellipse cx="260" cy="285" rx="120" ry="32" fill="url(#brainCenterGlow)" />

    <!-- Corner HUD Telemetry Markers -->
    <path d="M 40 50 L 40 30 L 60 30" fill="none" stroke="#00D4FF" stroke-width="2" opacity="0.8" />
    <path d="M 480 50 L 480 30 L 460 30" fill="none" stroke="#00D4FF" stroke-width="2" opacity="0.8" />
    <path d="M 40 330 L 40 350 L 60 350" fill="none" stroke="#00D4FF" stroke-width="2" opacity="0.8" />
    <path d="M 480 330 L 480 350 L 460 350" fill="none" stroke="#00D4FF" stroke-width="2" opacity="0.8" />

    <!-- HUD Telemetry Labels -->
    <text x="50" y="44" fill="#38BDF8" font-family="system-ui, sans-serif" font-size="10" font-weight="700" letter-spacing="1">MODALITY: AXIAL T1-MRI</text>
    <text x="50" y="58" fill="#94A3B8" font-family="system-ui, sans-serif" font-size="9" letter-spacing="0.5">MATRIX: 128×128 • RESAMPLED 224</text>
    <text x="355" y="44" fill="#38BDF8" font-family="system-ui, sans-serif" font-size="10" font-weight="700" letter-spacing="1">AI INFERENCE ENGINE</text>
    <text x="360" y="58" fill="#10B981" font-family="system-ui, sans-serif" font-size="9" font-weight="700" letter-spacing="0.5">● 3-MODEL CONSENSUS</text>

    <!-- 3D Layer 1: Cranial Depth Shadow Silhouette (Lower Layer) -->
    <g transform="translate(0, 16)" opacity="0.35">
      <path d="M 260 90 C 205 90, 170 120, 160 160 C 150 200, 165 245, 195 270 C 220 290, 245 295, 260 295 C 275 295, 300 290, 325 270 C 355 245, 370 200, 360 160 C 350 120, 315 90, 260 90 Z"
            fill="none" stroke="#1D4ED8" stroke-width="2.5" stroke-dasharray="5,3" />
    </g>

    <!-- 3D Layer 2: Primary Brain Cranial Outline (High Glow) -->
    <g filter="url(#neonGlow)">
      <!-- Left Hemisphere -->
      <path d="M 258 75
               C 215 75, 175 105, 165 145
               C 155 180, 165 215, 185 240
               C 195 252, 210 268, 230 276
               C 245 282, 255 284, 258 285
               C 256 250, 254 200, 255 140
               C 256 110, 257 90, 258 75 Z"
            fill="rgba(15, 23, 42, 0.45)" stroke="url(#cyberGrad)" stroke-width="2.2" />

      <!-- Right Hemisphere -->
      <path d="M 262 75
               C 305 75, 345 105, 355 145
               C 365 180, 355 215, 335 240
               C 325 252, 310 268, 290 276
               C 275 282, 265 284, 262 285
               C 264 250, 266 200, 265 140
               C 264 110, 263 90, 262 75 Z"
            fill="rgba(15, 23, 42, 0.45)" stroke="url(#cyberGrad)" stroke-width="2.2" />

      <!-- Longitudinal Fissure (Center Divider) -->
      <line x1="260" y1="75" x2="260" y2="285" stroke="#00D4FF" stroke-width="1.8" stroke-dasharray="7,3" opacity="0.75" />

      <!-- Ventricle Region Silhouette (Anatomical Region of Interest) -->
      <!-- Left Lateral Ventricle -->
      <path d="M 248 150 C 235 155, 230 175, 235 195 C 240 210, 248 215, 252 205 C 255 195, 254 170, 248 150 Z"
            fill="rgba(2, 132, 199, 0.25)" stroke="#38BDF8" stroke-width="1.4" />
      <!-- Right Lateral Ventricle -->
      <path d="M 272 150 C 285 155, 290 175, 285 195 C 280 210, 272 215, 268 205 C 265 195, 266 170, 272 150 Z"
            fill="rgba(2, 132, 199, 0.25)" stroke="#38BDF8" stroke-width="1.4" />

      <!-- Cortical Sulci & Gyri Holographic Ribbons -->
      <path d="M 185 125 C 205 135, 220 120, 240 135" fill="none" stroke="#60A5FA" stroke-width="1.2" opacity="0.7" />
      <path d="M 175 165 C 195 170, 210 160, 235 175" fill="none" stroke="#60A5FA" stroke-width="1.2" opacity="0.7" />
      <path d="M 185 210 C 205 205, 220 225, 245 220" fill="none" stroke="#60A5FA" stroke-width="1.2" opacity="0.7" />

      <path d="M 335 125 C 315 135, 300 120, 280 135" fill="none" stroke="#60A5FA" stroke-width="1.2" opacity="0.7" />
      <path d="M 345 165 C 325 170, 310 160, 285 175" fill="none" stroke="#60A5FA" stroke-width="1.2" opacity="0.7" />
      <path d="M 335 210 C 315 205, 300 225, 275 220" fill="none" stroke="#60A5FA" stroke-width="1.2" opacity="0.7" />
    </g>

    <!-- Neural Network Nodes & Synaptic Links -->
    <g class="neural-network-layer">
      <!-- Synaptic Lines -->
      <line x1="205" y1="130" x2="245" y2="160" stroke="#00D4FF" stroke-width="1" opacity="0.6" stroke-dasharray="3,2" />
      <line x1="245" y1="160" x2="275" y2="160" stroke="#00D4FF" stroke-width="1" opacity="0.8" />
      <line x1="275" y1="160" x2="315" y2="130" stroke="#00D4FF" stroke-width="1" opacity="0.6" stroke-dasharray="3,2" />
      <line x1="200" y1="190" x2="238" y2="190" stroke="#A855F7" stroke-width="1" opacity="0.7" />
      <line x1="282" y1="190" x2="320" y2="190" stroke="#A855F7" stroke-width="1" opacity="0.7" />
      <line x1="245" y1="160" x2="248" y2="215" stroke="#38BDF8" stroke-width="1" opacity="0.6" />
      <line x1="275" y1="160" x2="272" y2="215" stroke="#38BDF8" stroke-width="1" opacity="0.6" />
      <line x1="248" y1="215" x2="225" y2="250" stroke="#00D4FF" stroke-width="1" opacity="0.6" />
      <line x1="272" y1="215" x2="295" y2="250" stroke="#00D4FF" stroke-width="1" opacity="0.6" />

      <!-- Synaptic Pulsing Nodes -->
      <circle cx="205" cy="130" r="4.5" fill="#00D4FF" class="neural-node pulse-node" />
      <circle cx="315" cy="130" r="4.5" fill="#00D4FF" class="neural-node pulse-node" />
      <circle cx="245" cy="160" r="5.5" fill="#38BDF8" class="neural-node pulse-node" />
      <circle cx="275" cy="160" r="5.5" fill="#38BDF8" class="neural-node pulse-node" />
      <circle cx="200" cy="190" r="4.5" fill="#A855F7" class="neural-node pulse-node" />
      <circle cx="320" cy="190" r="4.5" fill="#A855F7" class="neural-node pulse-node" />
      <circle cx="248" cy="215" r="4.5" fill="#00D4FF" class="neural-node pulse-node" />
      <circle cx="272" cy="215" r="4.5" fill="#00D4FF" class="neural-node pulse-node" />
      <circle cx="225" cy="250" r="4" fill="#38BDF8" class="neural-node pulse-node" />
      <circle cx="295" cy="250" r="4" fill="#38BDF8" class="neural-node pulse-node" />
      <circle cx="260" cy="100" r="4" fill="#00D4FF" class="neural-node pulse-node" />
      <circle cx="260" cy="265" r="4" fill="#00D4FF" class="neural-node pulse-node" />
    </g>

    <!-- Animated Scanning Laser Line -->
    <g class="scan-laser-group">
      <rect x="150" y="60" width="220" height="4" fill="url(#scanBeamGrad)" class="scan-beam-rect" />
      <line x1="145" y1="62" x2="375" y2="62" stroke="#00D4FF" stroke-width="2" filter="url(#neonGlow)" class="scan-beam-line" />
    </g>

    <!-- Bottom Holographic Status Readout -->
    <rect x="160" y="325" width="200" height="28" rx="6" fill="rgba(15, 23, 42, 0.75)" stroke="#0284C7" stroke-width="1" />
    <text x="260" y="343" fill="#67E8F9" font-family="system-ui, sans-serif" font-size="11" font-weight="700" text-anchor="middle" letter-spacing="1.5">
      GRAD-CAM ATTRIBUTION READY
    </text>
  </svg>
</div>
"""
