# Documenting your code 

You have written some [readable code](01_codeStyle.md) that is 
[reasonably organized](03_organizeFiles.md) and, crucially, you 
[wrote a test for it](09_testing.md). There is only one thing left for you to do, 
to make sure your code can be easily used and adapted by future developers: 
You must document it!

To create some documentation, please navigate to `SIMULATeQCD/doc_src/`, 
where you can find its source code. Our source code is written 
using [Sphinx](https://www.sphinx-doc.org/en/master/), a python documentation 
generator, combined with [MyST](https://myst-parser.readthedocs.io/en/latest/), 
which is an extension for Sphinx supporting 
[Markdown](https://daringfireball.net/projects/markdown/). You can follow those 
links to learn more in detail about this syntax. With this documentation, we 
just want to show you how to get started.

Before you compile the documentation, make sure you
```shell
pip install -r requirements.txt
```
If you would like to use the convenience function `docs_src/build/auto_compile.sh`,
you will also need to install the shell command `entr`. 

## Documentation syntax

You can `# Title Your Page` like so. Each `## Heading` and `### Subheading` just 
requires more pound symbols #.

Links to `[other documentation](path_to_documentation.md)` and to 
`[webpages](https://www.web_link.org)` are accomplished in the same way.

To make a numbered list
```html
1. all you have to do
2. is start typing the numbers
3. like so
```
while an unordered list
```html
* is accomplished
* with asterisks
```

Boxes with short code snippets, like what you have seen above, are surrounded by the backtick \`. Use three backticks to set aside a block of code. You can specify the language for the code block so it correctly applies syntax highlighting. Possible languages include, but are not limited to, `shell`, `python`, and `html`. (Besides following the web links above, you can also see examples of the block code in action by looking into the documentation yourself.)

## Compilation

Once you have added a `.md` file for your file, incorporate it into the most appropriate overarching `.md` file. (For instance this one falls under `SIMULATeQCD/docs_src/02_contributions/contributions.md`.) You can see how you did by compiling your code using `SIMULATeQCD/docs_src/build/compile.sh`. If you want to see how it looks, you can open the `SIMULATeQCD/docs_src/build/index.html` file using your favorite browser. For example you could call `firefox index.html` from the command line inside the `build` folder. When it looks nice, you can [commit your changes](02_git.md).

## Some stylistic guidelines for documentation

SIMULATeQCD is, at the moment, managed only by a handful of rather busy scientists. Nevertheless we would like to have our code and documentation look professional and polished. This is difficult to achieve without someone explicitly going through and checking for consistency; the best we can do is probably a bottom-up approach, where we trust you to read these guidelines and try to keep them in mind. We will try to update this list as we think of more things.

1. Please use a spell checker.
2. Try to capitalize page titles and headings consistently. (For example on headings we are generally only capitalizing the first word, unless some grammatical rule would otherwise call for it.)
3. Set aside your code names and snippets in boxes.
4. Please link to other documentation such as research papers whenever you can.
5. Please include some easy examples to get started with what you wrote.
6. If you did some benchmarks, the documentation can be a good place to store that information.
7. Please try to keep your documentation up-to-date.
8. If you notice any mistakes in the documentation, please just make the change yourself, or if that is too complicated, please make an Issue. 

