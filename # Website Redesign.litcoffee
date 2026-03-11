# Website Redesign  

## Context  

This workflow is for REDESIGNING an existing website to match a new design reference.  
The existing site's structure, content, and copy must be preserved — only the visual  
layer changes. Do not invent new sections, remove existing content, or alter functionality.

---

## Workflow  

When the user provides a reference image (new design) and the existing site's code or URL:  

1. **Audit** the existing site. Identify all existing sections, components, content blocks,  
   and functionality. Document what must be preserved.  

2. **Map** the new design reference to the existing content structure. Note every visual  
   change required: colors, typography, spacing, layout shifts, component styles, etc.  

3. **Rebuild** the existing `index.html` using Tailwind CSS (via CDN), applying the new  
   design while keeping all original content intact. No external files unless requested.  

4. **Screenshot** the rebuilt page using Puppeteer (`npx puppeteer screenshot index.html  
   --fullpage`). If the page has distinct sections, capture those individually too.  

5. **Compare** your screenshot against the new design reference. Check for mismatches in:  
   - Spacing and padding (measure in px)  
   - Font sizes, weights, and line heights  
   - Colors (exact hex values)  
   - Alignment and positioning  
   - Border radii, shadows, and effects  
   - Responsive behavior  
   - Image/icon sizing and placement  

6. **Fix** every visual mismatch found. Do not alter content — only fix styling.  

7. **Re-screenshot** and compare again.  

8. **Repeat** steps 5–7 until the result is within ~2–3px of the reference everywhere.  

Do NOT stop after one pass. Always do at least 2 comparison rounds.  
Only stop when the user says so or when no visible differences remain.  

---

## Technical Defaults  

- Use Tailwind CSS via CDN (`<script src="https://cdn.tailwindcss.com"></script>`)  
- Use placeholder images from https://placehold.co/ when source assets aren't available  
- Preserve any real images that exist in the current site  
- Mobile-first responsive design  
- Single `index.html` file unless the user requests otherwise  

---

## Rules  

- **Preserve all existing content** — text, images, links, sections, and functionality  
- **Do not add** features, sections, or content not present in the existing site  
- **Do not remove** content even if it doesn't appear in the new design reference  
- Match the new design reference exactly on the visual layer  
- Do not "improve" the design beyond what the reference shows  
- If the user provides CSS classes, design tokens, or a style guide, use them verbatim  
- Keep code clean but don't over-abstract — inline Tailwind classes are fine  
- When comparing screenshots, be specific about what's wrong  
  (e.g., "heading is 32px but reference shows 24px", "gap between cards is 16px but should be 24px")