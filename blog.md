---
layout: default
---
### Recent Posts
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <span>({{ post.date | date: "%B %d, %Y" }})</span>
    </li>
  {% endfor %}
</ul>
<script>
  document.addEventListener('DOMContentLoaded', () => {
    const header = document.querySelector('h3');
    if (header && header.textContent.trim() === 'Recent Posts') {
      header.textContent = 'Blog';
    }
  });
</script>
