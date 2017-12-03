---
layout: post
title:  "images augmentation"
date:   2017-12-03 20:21:08 +0800
categories: image
---
# data preparing
 You can find the `water lily` images on [Google][google] or [Baidu][baidu] image, also you can take the photo about water lily by yourself. Here are some examples collected by me:

<img align="left" width="300" height="290" src="/assets/image/blue_1.jpg">
<img align="center" width="300" height="290" src="/assets/image/blue_4.jpg">
<img align="left" width="300" height="290" src="/assets/image/purple_1.jpg">
<img align="center" width="300" height="290" src="/assets/image/red_1.jpeg">

I have collected 20 pictures for each kinds of water lily, the first 15 for train and the others for test. The train set is inadequate so images augmentation should be implemented.

# images augmentation



You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight python %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[google]: https://www.google.com
[baidu]: https://www.baidu.com
[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
