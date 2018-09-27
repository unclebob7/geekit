---
layout: post
title: "juint test vs python unittest"
date: 2018-09-28 19:21:00
image: 'http://res.cloudinary.com/dm7h7e8xj/image/upload/c_scale,w_760/v1502757949/o-sombra_xyw4wq.jpg'
description: concise tutorial on junit test for android versus python unittest
category: 'language'
tags:
- java
- python
introduction: concise tutorial on junit test for android versus python unittest
---
In computer programming, <a href="https://en.wikipedia.org/wiki/Unit_testing">unit testing</a> is a software testing method by which individual units of source code, sets of one or more computer program modules together with associated control data, usage procedures, and operating procedures, are tested to determine whether they are fit for use.

> This tutorial offers you some basic concepts on junit5--java along with python unittest with respective IDE on **Android Studio** and **Pycharm**. 

## category of software testing
- **state testing** : validates if that code results in expecetd state
- **behavior testing** : validates if it executes the expected sequence of events
[Wiki definition](https://en.wikipedia.org/wiki/Category:Software_testing).

- **To bold text**, use `<strong>`.
- *To italicize text*, use `<em>`.
- Abbreviations, like <abbr title="HyperText Markup Langage">HTML</abbr> should use `<abbr>`, with an optional `title` attribute for the full phrase.
- Citations, like <cite>&mdash; Thiago Rossener</cite>, should use `<cite>`.
- <del>Deleted</del> text should use `<del>` and <ins>inserted</ins> text should use `<ins>`.
- Superscript <sup>text</sup> uses `<sup>` and subscript <sub>text</sub> uses `<sub>`.

Most of these elements are styled by browsers with few modifications on our part.

## testing terminology
- **code/application under test** : explicit as it is , the sequence(code section) you settle under a test
- **test fixture** : specified statement you made with specific arguments(parameter) for outcome , a fixed state in code which is tested used as input for a test. 
*Another way to describe this is a test precondition.*

## Testing with JUnit4
### notation
- `@Test` notation indicates the section(function) serves testing function
- `public void function_name()` is an automatically generated function sharing same name for your **code under test** 

Here Bob offers a visibly easy `MyClass` with 2 functions `multiply(int x , int y)` and `add(int x , int y)`
```js
public class MyClass {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        int a=0;
        int b=0;
        int result = 0;

        System.out.println("enter figure for multiple sequence:");
        a = input.nextInt();
        b = input.nextInt();
        result = multiply(a,b);

        System.out.println(result);
    }

    //in use
    public static int multiply(int x , int y) {
        return x*y;
    }
    //not in use
    public static int add(int x , int y) {
        return x+y;
    }
}
```
**BTW** always remember to settle your function static for your main-function entrance to avoid memory leak.

### Instantialize a test_class

![library_to_choose](https://res.cloudinary.com/dn18ydekv/image/upload/v1538049787/choose_library.png)

* Praesent commodo cursus magna, vel scelerisque nisl consectetur et.
* Donec id elit non mi porta gravida at eget metus.
* Nulla vitae elit libero, a pharetra augue.

Donec ullamcorper nulla non metus auctor fringilla. Nulla vitae elit libero, a pharetra augue.

1. Vestibulum id ligula porta felis euismod semper.
2. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.
3. Maecenas sed diam eget risus varius blandit sit amet non magna.

Cras mattis consectetur purus sit amet fermentum. Sed posuere consectetur est at lobortis.

Integer posuere erat a ante venenatis dapibus posuere velit aliquet. Morbi leo risus, porta ac consectetur ac, vestibulum at eros. Nullam quis risus eget urna mollis ornare vel eu leo.

## Images

Quisque consequat sapien eget quam rhoncus, sit amet laoreet diam tempus. Aliquam aliquam metus erat, a pulvinar turpis suscipit at.

![placeholder](https://placehold.it/800x400 "Large example image")
![placeholder](https://placehold.it/400x200 "Medium example image")
![placeholder](https://placehold.it/200x200 "Small example image")

## Tables

Aenean lacinia bibendum nulla sed consectetur. Lorem ipsum dolor sit amet, consectetur adipiscing elit.

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Upvotes</th>
      <th>Downvotes</th>
    </tr>
  </thead>
  <tfoot>
    <tr>
      <td>Totals</td>
      <td>21</td>
      <td>23</td>
    </tr>
  </tfoot>
  <tbody>
    <tr>
      <td>Alice</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <td>Bob</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Charlie</td>
      <td>7</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

Nullam id dolor id nibh ultricies vehicula ut id elit. Sed posuere consectetur est at lobortis. Nullam quis risus eget urna mollis ornare vel eu leo.

-----

Want to see something else added? <a href="https://github.com/poole/poole/issues/new">Open an issue.</a>















