---
layout: get_started
title: PyTorch 버전 호환성
permalink: /get-started/compatibility/
featured-img: "assets/images/featured-img-pytorch-2.png"
background-class: get-started-background
body-class: get-started
order: 5
published: true
---

## PyTorch 버전 호환성

PyTorch 및 Domain APIs의 버전 호환성을 정리하였습니다. \
각 PyTorch 버전별로 호환되는 Domain APIs 및 지원 환경(Python 및 CUDA/ROCm 버전 등)을 확인해보세요. \
\
💡 표를 좌/우 스크롤하여 모든 정보를 확인할 수 있습니다.

<table class="table table-striped table-hover version-table">
  <thead class="thead-dark">
    <tr>
      <th scope="col"><strong>PyTorch</strong></th>
      <th scope="col"><strong>Release Date</strong></th>
      <th scope="col"><strong><a href="https://docs.pytorch.org/vision/stable/" target="_blank">torchvision</a></strong></th>
      <th scope="col"><strong><a href="https://docs.pytorch.org/audio/stable/" target="_blank">torchaudio</a></strong></th>
      <th scope="col"><strong><a href="https://docs.pytorch.org/text/stable/" target="_blank">torchtext</a></strong></th>
      <th scope="col"><strong><a href="https://meta-pytorch.org/data/beta/" target="_blank">torchdata</a></strong></th>
      <th scope="col"><strong><a href="https://docs.pytorch.org/torchcodec/stable/" target="_blank">torchcodec</a></strong></th>
      <th scope="col"><strong>Python</strong></th>
      <th scope="col"><strong>CUDA</strong></th>
      <th scope="col"><strong>ROCm</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.13.0" target="_blank">2.13.0</a></th>
      <td>2026/07/08</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.28.0" target="_blank">0.28.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.14.0" target="_blank">0.14.0</a></td>
      <td><a href="https://pypi.org/project/torch/2.13.0/" target="_blank">>=3.10, <=3.14</a></td>
      <td>13.2; 13.0; 12.6</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.12.1" target="_blank">2.12.1</a></th>
      <td>2026/06/18</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.27.1" target="_blank">0.27.1</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.13.0" target="_blank">0.13.0</a></td>
      <td><a href="https://pypi.org/project/torch/2.12.1/" target="_blank">>=3.10, <=3.14</a></td>
      <td>13.2; 13.0; 12.6</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.12.0" target="_blank">2.12.0</a></th>
      <td>2026/05/13</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.27.0" target="_blank">0.27.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.12.0" target="_blank">0.12.0</a></td>
      <td><a href="https://pypi.org/project/torch/2.12.0/" target="_blank">>=3.10, <=3.14</a></td>
      <td>13.2; 13.0; 12.6</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.11.0" target="_blank">2.11.0</a></th>
      <td>2026/03/23</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.26.0" target="_blank">0.26.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.11.0" target="_blank">2.11.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.11.0" target="_blank">0.11.0</a></td>
      <td><a href="https://pypi.org/project/torch/2.11.0/" target="_blank">>=3.10, <=3.14</a></td>
      <td>13.0; 12.8; 12.6</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.10.0" target="_blank">2.10.0</a></th>
      <td>2026/01/21</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.25.0" target="_blank">0.25.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.10.0" target="_blank">2.10.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.10.0" target="_blank">0.10.0</a></td>
      <td><a href="https://pypi.org/project/torch/2.10.0/" target="_blank">>=3.10, <=3.14</a></td>
      <td>13.0; 12.8; 12.6</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.9.1" target="_blank">2.9.1</a></th>
      <td>2025/11/12</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.24.1" target="_blank">0.24.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.9.1" target="_blank">2.9.1</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.9.1" target="_blank">0.9.1</a></td>
      <td><a href="https://pypi.org/project/torch/2.9.1/" target="_blank">>=3.10, <=3.14</a></td>
      <td>13.0; 12.8; 12.6</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.9.0" target="_blank">2.9.0</a></th>
      <td>2025/10/15</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.24.0" target="_blank">0.24.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.9.0" target="_blank">2.9.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.8.0" target="_blank">0.8.0</a></td>
      <td><a href="https://pypi.org/project/torch/2.9.0/" target="_blank">>=3.10, <=3.14</a></td>
      <td>13.0; 12.8; 12.6</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.8.0" target="_blank">2.8.0</a></th>
      <td>2025/08/06</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.23.0" target="_blank">0.23.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.8.0" target="_blank">2.8.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.7.0" target="_blank">0.7.0</a></td>
      <td><a href="https://pypi.org/project/torch/2.8.0/" target="_blank">>=3.9, <=3.13</a></td>
      <td>12.9; 12.8; 12.6; 11.8</td>
      <td>6.3</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.7.1" target="_blank">2.7.1</a></th>
      <td>2025/06/04</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.22.1" target="_blank">0.22.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.7.1" target="_blank">2.7.1</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.5.0" target="_blank">0.5.0</a></td>
      <td><a href="https://pypi.org/project/torch/2.7.1/" target="_blank">>=3.9, <=3.13</a></td>
      <td>12.8; 12.6; 11.8</td>
      <td>6.3</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.7.0" target="_blank">2.7.0</a></th>
      <td>2025/04/23</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.22.0" target="_blank">0.22.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.7.0" target="_blank">2.7.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.4.0" target="_blank">0.4.0</a></td>
      <td><a href="https://pypi.org/project/torch/2.7.0/" target="_blank">>=3.9, <=3.13</a></td>
      <td>12.8; 12.6; 11.8</td>
      <td>6.3</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.6.0" target="_blank">2.6.0</a></th>
      <td>2025/01/29</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.21.0" target="_blank">0.21.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.6.0" target="_blank">2.6.0</a></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.11.0" target="_blank">0.11.0</a></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.2.1" target="_blank">0.2.1</a></td>
      <td><a href="https://pypi.org/project/torch/2.6.0/" target="_blank">>=3.9, <=3.13</a></td>
      <td>12.6; 12.4; 11.8</td>
      <td>6.2.4; 6.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.5.1" target="_blank">2.5.1</a></th>
      <td>2024/10/29</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.20.1" target="_blank">0.20.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.5.1" target="_blank">2.5.1</a></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.10.1" target="_blank">0.10.1</a></td>
      <td></td>
      <td><a href="https://pypi.org/project/torch/2.5.1/" target="_blank">>=3.8, <=3.13</a></td>
      <td>12.4; 12.1; 11.8</td>
      <td>6.2; 6.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.5.0" target="_blank">2.5.0</a></th>
      <td>2024/10/17</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.20.0" target="_blank">0.20.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.5.0" target="_blank">2.5.0</a></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.10.1" target="_blank">0.10.1</a></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.1.1" target="_blank">0.1.1</a></td>
      <td><a href="https://pypi.org/project/torch/2.5.0/" target="_blank">>=3.8, <=3.12</a></td>
      <td>12.4; 12.1; 11.8</td>
      <td>6.2; 6.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.4.1" target="_blank">2.4.1</a></th>
      <td>2024/09/04</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.19.1" target="_blank">0.19.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.4.1" target="_blank">2.4.1</a></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.8.0" target="_blank">0.8.0</a></td>
      <td></td>
      <td><a href="https://pypi.org/project/torch/2.4.1/" target="_blank">>=3.8, <=3.12</a></td>
      <td>12.4; 12.1; 11.8</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.4.0" target="_blank">2.4.0</a></th>
      <td>2024/07/24</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.19.0" target="_blank">0.19.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.4.0" target="_blank">2.4.0</a></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.8.0" target="_blank">0.8.0</a></td>
      <td><a href="https://github.com/meta-pytorch/torchcodec/releases/tag/v0.0.3" target="_blank">0.0.3</a></td>
      <td><a href="https://pypi.org/project/torch/2.4.0/" target="_blank">>=3.8, <=3.12</a></td>
      <td>12.4; 12.1; 11.8</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.3.1" target="_blank">2.3.1</a></th>
      <td>2024/06/05</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.18.1" target="_blank">0.18.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.3.1" target="_blank">2.3.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.18.0" target="_blank">0.18.0</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/2.3.1/" target="_blank">>=3.8, <=3.12</a></td>
      <td>12.1; 11.8</td>
      <td>6.0; 5.7</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.3.0" target="_blank">2.3.0</a></th>
      <td>2024/04/24</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.18.0" target="_blank">0.18.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.3.0" target="_blank">2.3.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.18.0" target="_blank">0.18.0</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/2.3.0/" target="_blank">>=3.8, <=3.12</a></td>
      <td>12.1; 11.8</td>
      <td>6.0; 5.7</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.2.2" target="_blank">2.2.2</a></th>
      <td>2024/03/28</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.17.2" target="_blank">0.17.2</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.2.2" target="_blank">2.2.2</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.17.2" target="_blank">0.17.2</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/2.2.2/" target="_blank">>=3.8, <=3.12</a></td>
      <td>12.1; 11.8</td>
      <td>5.7; 5.6</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.2.1" target="_blank">2.2.1</a></th>
      <td>2024/02/22</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.17.1" target="_blank">0.17.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.2.1" target="_blank">2.2.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.17.1" target="_blank">0.17.1</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/2.2.1/" target="_blank">>=3.8, <=3.12</a></td>
      <td>12.1; 11.8</td>
      <td>5.7; 5.6</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.2.0" target="_blank">2.2.0</a></th>
      <td>2024/01/30</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.17.0" target="_blank">0.17.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.2.0" target="_blank">2.2.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.17.0" target="_blank">0.17.0</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/2.2.0/" target="_blank">>=3.8, <=3.12</a></td>
      <td>12.1; 11.8</td>
      <td>5.7; 5.6</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.1.2" target="_blank">2.1.2</a></th>
      <td>2023/12/15</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.16.2" target="_blank">0.16.2</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.1.2" target="_blank">2.1.2</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.16.2" target="_blank">0.16.2</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/2.1.2/" target="_blank">>=3.8, <=3.11</a></td>
      <td>12.1; 11.8</td>
      <td>5.6; 5.5</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.1.1" target="_blank">2.1.1</a></th>
      <td>2023/11/15</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.16.1" target="_blank">0.16.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.1.1" target="_blank">2.1.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.16.1" target="_blank">0.16.1</a></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.7.1" target="_blank">0.7.1</a></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/2.1.1/" target="_blank">>=3.8, <=3.11</a></td>
      <td>12.1; 11.8</td>
      <td>5.6; 5.5</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.1.0" target="_blank">2.1.0</a></th>
      <td>2023/10/04</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.16.0" target="_blank">0.16.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.1.0" target="_blank">2.1.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.16.0" target="_blank">0.16.0</a></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.7.0" target="_blank">0.7.0</a></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/2.1.0/" target="_blank">>=3.8, <=3.11</a></td>
      <td>12.1; 11.8</td>
      <td>5.6; 5.5</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.0.1" target="_blank">2.0.1</a></th>
      <td>2023/05/09</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.15.2" target="_blank">0.15.2</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.0.2" target="_blank">2.0.2</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.15.2" target="_blank">0.15.2</a></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.6.1" target="_blank">0.6.1</a></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/2.0.1/" target="_blank">>=3.8, <=3.11</a></td>
      <td>11.8; 11.7</td>
      <td>5.4.2; 5.3</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v2.0.0" target="_blank">2.0.0</a></th>
      <td>2023/03/15</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.15.1" target="_blank">0.15.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v2.0.1" target="_blank">2.0.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.15.1" target="_blank">0.15.1</a></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.6.0" target="_blank">0.6.0</a></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/2.0.0/" target="_blank">>=3.8, <=3.11</a></td>
      <td>11.8; 11.7</td>
      <td>5.4.2; 5.3</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.13.1" target="_blank">1.13.1</a></th>
      <td>2022/12/16</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.14.1" target="_blank">0.14.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.13.1" target="_blank">0.13.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.14.1" target="_blank">0.14.1</a></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.5.1" target="_blank">0.5.1</a></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/1.13.1/" target="_blank">>=3.7, <=3.11</a></td>
      <td>11.7; 11.6</td>
      <td>5.2; 5.1.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.13.0" target="_blank">1.13.0</a></th>
      <td>2022/10/28</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.14.0" target="_blank">0.14.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.13.0" target="_blank">0.13.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.14.0" target="_blank">0.14.0</a></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.5.0" target="_blank">0.5.0</a></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/1.13.0/" target="_blank">>=3.7, <=3.11</a></td>
      <td>11.7; 11.6</td>
      <td>5.2; 5.1.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.12.1" target="_blank">1.12.1</a></th>
      <td>2022/08/05</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.13.1" target="_blank">0.13.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.12.1" target="_blank">0.12.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.13.1" target="_blank">0.13.1</a></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.4.1" target="_blank">0.4.1</a></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/1.12.1/" target="_blank">>=3.7, <=3.10</a></td>
      <td>11.6; 11.3; 10.2</td>
      <td>5.1.1; 5.0</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.12.0" target="_blank">1.12.0</a></th>
      <td>2022/06/28</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.13.0" target="_blank">0.13.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.12.0" target="_blank">0.12.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.13.0" target="_blank">0.13.0</a></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.4.0" target="_blank">0.4.0</a></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/1.12.0/" target="_blank">>=3.7, <=3.10</a></td>
      <td>11.6; 11.3; 10.2</td>
      <td>5.1.1; 5.0</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.11.0" target="_blank">1.11.0</a></th>
      <td>2022/03/10</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.12.0" target="_blank">0.12.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.11.0" target="_blank">0.11.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.12.0" target="_blank">0.12.0</a></td>
      <td><a href="https://github.com/pytorch/data/releases/tag/v0.3.0" target="_blank">0.3.0</a></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/1.11.0/" target="_blank">>=3.7, <=3.10</a></td>
      <td>11.5; 11.3; 10.2</td>
      <td>4.5.2; 4.3.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.10.2" target="_blank">1.10.2</a></th>
      <td>2022/01/27</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.11.3" target="_blank">0.11.3</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.10.1" target="_blank">0.10.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.11.2" target="_blank">0.11.2</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/1.10.2/" target="_blank">>=3.6, <=3.9</a></td>
      <td>11.3; 11.1; 10.2</td>
      <td>4.2; 4.1; 4.0.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.10.1" target="_blank">1.10.1</a></th>
      <td>2021/12/15</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.11.2" target="_blank">0.11.2</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.10.1" target="_blank">0.10.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.11.1" target="_blank">0.11.1</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/1.10.1/" target="_blank">>=3.6, <=3.9</a></td>
      <td>11.3; 11.1; 10.2</td>
      <td>4.2; 4.1; 4.0.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.10.0" target="_blank">1.10.0</a></th>
      <td>2021/10/21</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.11.1" target="_blank">0.11.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.10.0" target="_blank">0.10.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.11.0" target="_blank">0.11.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><a href="https://pypi.org/project/torch/1.10.0/" target="_blank">>=3.6, <=3.9</a></td>
      <td>11.3; 11.1; 10.2</td>
      <td>4.2; 4.1; 4.0.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.9.1" target="_blank">1.9.1</a></th>
      <td>2021/09/22</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.10.1" target="_blank">0.10.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.9.1" target="_blank">0.9.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.10.1" target="_blank">0.10.1</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>11.1; 10.2</td>
      <td>4.2; 4.1; 4.0.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.9.0" target="_blank">1.9.0</a></th>
      <td>2021/06/15</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.10.0" target="_blank">0.10.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.9.0" target="_blank">0.9.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.10.0" target="_blank">0.10.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>11.1; 10.2</td>
      <td>4.2; 4.1; 4.0.1</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.8.2" target="_blank">1.8.2</a></th>
      <td>2021/08/17</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.9.1" target="_blank">0.9.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.8.2" target="_blank">0.8.2</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.9.1" target="_blank">0.9.1</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.8.1" target="_blank">1.8.1</a></th>
      <td>2021/03/25</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.9.1" target="_blank">0.9.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.8.1" target="_blank">0.8.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.9.1" target="_blank">0.9.1</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>11.1; 10.2; 10.1</td>
      <td>4.0.1; 3.10</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.8.0" target="_blank">1.8.0</a></th>
      <td>2021/03/04</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.9.0" target="_blank">0.9.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.8.0" target="_blank">0.8.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.9.0" target="_blank">0.9.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>11.1; 10.2; 10.1</td>
      <td>4.0.1; 3.10</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.7.1" target="_blank">1.7.1</a></th>
      <td>2020/12/10</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.8.2" target="_blank">0.8.2</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.7.2" target="_blank">0.7.2</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.8.1" target="_blank">0.8.1</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>11.0; 10.2; 10.1; 9.2</td>
      <td>3.8; 3.7</td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.7.0" target="_blank">1.7.0</a></th>
      <td>2020/10/27</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.8.0" target="_blank">0.8.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.7.0" target="_blank">0.7.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.8.0" target="_blank">0.8.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>11.0; 10.2; 10.1; 9.2</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.6.0" target="_blank">1.6.0</a></th>
      <td>2020/07/28</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.7.0" target="_blank">0.7.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.6.0" target="_blank">0.6.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/0.7.0" target="_blank">0.7.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>10.2; 10.1; 9.2</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.5.1" target="_blank">1.5.1</a></th>
      <td>2020/06/18</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.6.1" target="_blank">0.6.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.5.1" target="_blank">0.5.1</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/0.6.0" target="_blank">0.6.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>10.2; 10.1; 9.2</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.5.0" target="_blank">1.5.0</a></th>
      <td>2020/04/21</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.6.0" target="_blank">0.6.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.5.0" target="_blank">0.5.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/0.6.0" target="_blank">0.6.0</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>10.2; 10.1; 9.2</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.4.0" target="_blank">1.4.0</a></th>
      <td>2020/01/16</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.5.0" target="_blank">0.5.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.4.0" target="_blank">0.4.0</a></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/0.5.0" target="_blank">0.5.0</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>10.1; 10.0; 9.2</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.3.1" target="_blank">1.3.1</a></th>
      <td>2019/11/07</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.4.2" target="_blank">0.4.2</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.3.2" target="_blank">0.3.2</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>10.1; 10.0; 9.2</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.3.0" target="_blank">1.3.0</a></th>
      <td>2019/10/10</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.4.1" target="_blank">0.4.1</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.3.1" target="_blank">0.3.1</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>10.1; 10.0; 9.2</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.2.0" target="_blank">1.2.0</a></th>
      <td>2019/08/08</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.4.0" target="_blank">0.4.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.3.0" target="_blank">0.3.0</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>10.0; 9.2</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.1.0" target="_blank">1.1.0</a></th>
      <td>2019/05/01</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.3.0" target="_blank">0.3.0</a></td>
      <td><a href="https://github.com/pytorch/audio/releases/tag/v0.3.2" target="_blank">0.2.0</a></td>
      <td></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>10.0; 9.0</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.0.1" target="_blank">1.0.1</a></th>
      <td>2019/02/07</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.2.2" target="_blank">0.2.2</a></td>
      <td><em>—</em></td>
      <td></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>10.0; 9.0; 8.0</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v1.0.0" target="_blank">1.0.0</a></th>
      <td>2018/12/07</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.2.1" target="_blank">0.2.1</a></td>
      <td><em>—</em></td>
      <td></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>10.0; 9.0; 8.0</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.4.1" target="_blank">0.4.1</a></th>
      <td>2018/07/26</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.2.1" target="_blank">0.2.1</a></td>
      <td><em>—</em></td>
      <td></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>9.2; 8.0</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.4.0" target="_blank">0.4.0</a></th>
      <td>2018/04/24</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.2.1" target="_blank">0.2.1</a></td>
      <td><em>—</em></td>
      <td></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>9.1; 9.0; 8.0</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.3.1" target="_blank">0.3.1</a></th>
      <td>2018/02/14</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.2.0" target="_blank">0.2.0</a></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.2.3" target="_blank">0.2.3</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>9.1; 9.0; 8.0</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.3.0" target="_blank">0.3.0</a></th>
      <td>2017/12/05</td>
      <td><a href="https://github.com/pytorch/vision/releases/tag/v0.2.0" target="_blank">0.2.0</a></td>
      <td><em>—</em></td>
      <td><a href="https://github.com/pytorch/text/releases/tag/v0.2.3" target="_blank">0.2.3</a></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>9.0; 8.0; 7.5</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.2.0" target="_blank">0.2.0</a></th>
      <td>2017/08/28</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>8.0; 7.5</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.12" target="_blank">0.1.12</a></th>
      <td>2017/05/02</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>8.0; 7.5</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.11" target="_blank">0.1.11</a></th>
      <td>2017/03/31</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>8.0; 7.5</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.10" target="_blank">0.1.10</a></th>
      <td>2017/03/15</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>8.0; 7.5</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.9" target="_blank">0.1.9</a></th>
      <td>2017/02/24</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>8.0; 7.5</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.8" target="_blank">0.1.8</a></th>
      <td>2017/02/05</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>8.0; 7.5</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.7" target="_blank">0.1.7</a></th>
      <td>2017/02/02</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>8.0; 7.5</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.6" target="_blank">0.1.6</a></th>
      <td>2017/02/02</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td>8.0; 7.5</td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.5" target="_blank">0.1.5</a></th>
      <td>2016/11/18</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.4" target="_blank">0.1.4</a></th>
      <td>2016/10/03</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.3" target="_blank">0.1.3</a></th>
      <td>2016/09/16</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.2" target="_blank">0.1.2</a></th>
      <td>2016/09/01</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
    </tr>
    <tr>
      <th scope="row"><a href="https://github.com/pytorch/pytorch/releases/tag/v0.1.1" target="_blank">0.1.1</a></th>
      <td>2016/09/01</td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
      <td><em>—</em></td>
    </tr>
  </tbody>
</table>


## Release Compatibility Matrix

PyTorch 릴리즈별 지원 환경(Python, C++, CUDA, ROCm) 호환성 매트릭스입니다.

<table class="table table-striped table-hover version-table">
  <thead class="thead-dark">
    <tr>
      <th scope="col"><strong>PyTorch</strong></th>
      <th scope="col"><strong>Python</strong></th>
      <th scope="col"><strong>C++</strong></th>
      <th scope="col"><strong>Stable CUDA</strong></th>
      <th scope="col"><strong>Experimental CUDA</strong></th>
      <th scope="col"><strong>Stable ROCm</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">2.13</th>
      <td>>=3.10, <=(3.15, 3.15t experimental)</td>
      <td>C++20</td>
      <td>CUDA 12.6 (CUDNN 9.10.2.21) (NCCL 2.29.3), CUDA 13.0 (CUDNN 9.20.0.48) (NCCL 2.29.7)</td>
      <td>CUDA 13.2 (CUDNN 9.20.0.48) (NCCL 2.29.7)</td>
      <td>ROCm 7.2</td>
    </tr>
    <tr>
      <th scope="row">2.12</th>
      <td>>=3.10, <=(3.14, 3.14t experimental)</td>
      <td>C++17</td>
      <td>CUDA 12.6 (CUDNN 9.10.2.21), CUDA 13.0 (CUDNN 9.20.0.48)</td>
      <td>CUDA 13.2 (CUDNN 9.20.0.48)</td>
      <td>ROCm 7.2</td>
    </tr>
    <tr>
      <th scope="row">2.11</th>
      <td>>=3.10, <=(3.14, 3.14t experimental)</td>
      <td>C++17</td>
      <td>CUDA 12.6 (CUDNN 9.10.2.21), CUDA 12.8 (CUDNN 9.17.1.4), CUDA 13.0 (CUDNN 9.17.1.4)</td>
      <td>—</td>
      <td>ROCm 7.2</td>
    </tr>
    <tr>
      <th scope="row">2.10</th>
      <td>>=3.10, <=(3.14, 3.14t experimental)</td>
      <td>C++17</td>
      <td>CUDA 12.6 (CUDNN 9.10.2.21), CUDA 12.8 (CUDNN 9.10.2.21)</td>
      <td>CUDA 13.0 (CUDNN 9.15.1.9)</td>
      <td>ROCm 7.1</td>
    </tr>
    <tr>
      <th scope="row">2.9</th>
      <td>>=3.10, <=(3.14, 3.14t experimental)</td>
      <td>C++17</td>
      <td>CUDA 12.6 (CUDNN 9.10.2.21), CUDA 12.8 (CUDNN 9.10.2.21)</td>
      <td>CUDA 13.0 (CUDNN 9.13.0.50)</td>
      <td>ROCm 6.4</td>
    </tr>
    <tr>
      <th scope="row">2.8</th>
      <td>>=3.9, <=3.13, (3.13t experimental)</td>
      <td>C++17</td>
      <td>CUDA 12.6 (CUDNN 9.10.2.21), CUDA 12.8 (CUDNN 9.10.2.21)</td>
      <td>CUDA 12.9 (CUDNN 9.10.2.21)</td>
      <td>ROCm 6.4</td>
    </tr>
    <tr>
      <th scope="row">2.7</th>
      <td>>=3.9, <=3.13, (3.13t experimental)</td>
      <td>C++17</td>
      <td>CUDA 11.8 (CUDNN 9.1.0.70), CUDA 12.6 (CUDNN 9.5.1.17)</td>
      <td>CUDA 12.8 (CUDNN 9.7.1.26)</td>
      <td>ROCm 6.3</td>
    </tr>
    <tr>
      <th scope="row">2.6</th>
      <td>>=3.9, <=3.13, (3.13t experimental)</td>
      <td>C++17</td>
      <td>CUDA 11.8, CUDA 12.4 (CUDNN 9.1.0.70)</td>
      <td>CUDA 12.6 (CUDNN 9.5.1.17)</td>
      <td>ROCm 6.2.4</td>
    </tr>
    <tr>
      <th scope="row">2.5</th>
      <td>>=3.9, <=3.12, (3.13 experimental)</td>
      <td>C++17</td>
      <td>CUDA 11.8, CUDA 12.1, CUDA 12.4, CUDNN 9.1.0.70</td>
      <td>—</td>
      <td>ROCm 6.2</td>
    </tr>
    <tr>
      <th scope="row">2.4</th>
      <td>>=3.8, <=3.12</td>
      <td>C++17</td>
      <td>CUDA 11.8, CUDA 12.1, CUDNN 9.1.0.70</td>
      <td>CUDA 12.4, CUDNN 9.1.0.70</td>
      <td>ROCm 6.1</td>
    </tr>
    <tr>
      <th scope="row">2.3</th>
      <td>>=3.8, <=3.11, (3.12 experimental)</td>
      <td>C++17</td>
      <td>CUDA 11.8, CUDNN 8.7.0.84</td>
      <td>CUDA 12.1, CUDNN 8.9.2.26</td>
      <td>ROCm 6.0</td>
    </tr>
    <tr>
      <th scope="row">2.2</th>
      <td>>=3.8, <=3.11, (3.12 experimental)</td>
      <td>C++17</td>
      <td>CUDA 11.8, CUDNN 8.7.0.84</td>
      <td>CUDA 12.1, CUDNN 8.9.2.26</td>
      <td>ROCm 5.7</td>
    </tr>
    <tr>
      <th scope="row">2.1</th>
      <td>>=3.8, <=3.11</td>
      <td>C++17</td>
      <td>CUDA 11.8, CUDNN 8.7.0.84</td>
      <td>CUDA 12.1, CUDNN 8.9.2.26</td>
      <td>ROCm 5.6</td>
    </tr>
    <tr>
      <th scope="row">2.0</th>
      <td>>=3.8, <=3.11</td>
      <td>C++14</td>
      <td>CUDA 11.7, CUDNN 8.5.0.96</td>
      <td>CUDA 11.8, CUDNN 8.7.0.84</td>
      <td>ROCm 5.4</td>
    </tr>
    <tr>
      <th scope="row">1.13</th>
      <td>>=3.7, <=3.10</td>
      <td>C++14</td>
      <td>CUDA 11.6, CUDNN 8.3.2.44</td>
      <td>CUDA 11.7, CUDNN 8.5.0.96</td>
      <td>ROCm 5.2</td>
    </tr>
    <tr>
      <th scope="row">1.12</th>
      <td>>=3.7, <=3.10</td>
      <td>C++14</td>
      <td>CUDA 11.3, CUDNN 8.3.2.44</td>
      <td>CUDA 11.6, CUDNN 8.3.2.44</td>
      <td>ROCm 5.0</td>
    </tr>
  </tbody>
</table>


## PyTorch CUDA Support Matrix

PyTorch 2.11 릴리즈에서 지원하는 CUDA 아키텍처 정보입니다.

### Linux x86 및 Windows

<table class="table table-striped table-hover version-table">
  <thead class="thead-dark">
    <tr>
      <th scope="col"><strong>CUDA</strong></th>
      <th scope="col"><strong>지원 아키텍처</strong></th>
      <th scope="col"><strong>비고</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>12.6.3</td>
      <td>Maxwell(5.0), Pascal(6.0), Volta(7.0), Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0)</td>
      <td></td>
    </tr>
    <tr>
      <td>12.8.1</td>
      <td>Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0), Blackwell(10.0, 12.0)</td>
      <td></td>
    </tr>
    <tr>
      <td>13.0.2</td>
      <td>Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0), Blackwell(10.0, 12.0+PTX)</td>
      <td>+PTX는 Linux 빌드에서만 사용 가능</td>
    </tr>
  </tbody>
</table>

### Linux aarch64

<table class="table table-striped table-hover version-table">
  <thead class="thead-dark">
    <tr>
      <th scope="col"><strong>CUDA</strong></th>
      <th scope="col"><strong>지원 아키텍처</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>12.6.3</td>
      <td>Ampere(8.0), Hopper(9.0)</td>
    </tr>
    <tr>
      <td>12.8.1</td>
      <td>Ampere(8.0), Hopper(9.0), Blackwell(10.0, 12.0)</td>
    </tr>
    <tr>
      <td>13.0.2</td>
      <td>Ampere(8.0), Hopper(9.0), Blackwell(10.0, 11.0, 12.0+PTX)</td>
    </tr>
  </tbody>
</table>


## 참고 (Notes)

### 버전 정보 참고

- 릴리즈 날짜는 PyTorch 저장소의 Releases 메뉴를 참고하였습니다. 모든 날짜는 UTC 기준으로 표기하였습니다.
- PyTorch 버전별 호환 Python 버전 정보는 <a href="https://pypi.org/project/torch/" target="_blank">PyTorch 패키지의 PyPI 저장소</a>를 참고하였습니다.
- CUDA 및 ROCm 버전은 <a href="/get-started/previous-versions/">이전 버전의 PyTorch</a> 문서 및 <a href="https://download.pytorch.org/whl/torch_stable.html" target="_blank">Stable Wheel 목록</a>을 참고하였습니다.

### Domain Library 참고
- torchtext는 0.18.0 버전(PyTorch 2.3.0 호환)을 마지막으로 <a href="https://github.com/pytorch/text" target="_blank">개발이 중단</a>되었습니다.


## 출처 (References)

이 문서의 정보들은 다음 문서들을 참고하였습니다:

### PyTorch Version & Release Data

- <a href="https://github.com/pytorch/pytorch/releases" target="_blank">PyTorch 저장소 Releases</a>
- <a href="https://pypi.org/project/torch/" target="_blank">PyTorch PyPI 페이지</a>
- <a href="/get-started/previous-versions/">이전 버전의 PyTorch</a>
- <a href="https://download.pytorch.org/whl/torch_stable.html" target="_blank">PyTorch Stable Wheel 목록</a>

### Domain Libraries

- torchvision: <a href="https://pypi.org/project/torchvision/" target="_blank">PyPI 페이지</a>
- torchaudio: <a href="https://docs.pytorch.org/audio/main/installation.html" target="_blank">공식 설치 문서 (&gt; 0.4.0)</a>, <a href="https://github.com/pytorch/audio/releases" target="_blank">저장소 Releases (&lt;= 0.4.0)</a>
- torchtext: <a href="https://github.com/pytorch/text?tab=readme-ov-file#installation" target="_blank">저장소 README</a>
- torchdata: <a href="https://pypi.org/project/torchdata/" target="_blank">PyPI 페이지</a>
- torchcodec: <a href="https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec" target="_blank">저장소 README</a>

### Others

- <a href="https://github.com/pytorch/pytorch/wiki/PyTorch-Versions" target="_blank">PyTorch Versions (공식 위키)</a>
- <a href="/get-started/locally/">PyTorch 시작하기</a>


<div class="alert alert-info" role="alert">
  <strong>📝 잘못된 내용을 발견하셨나요?</strong> <a href="https://github.com/PyTorchKorea/pytorch.kr/issues/new?template=documentation-content.md" target="_blank" class="alert-link">PyTorchKR 홈페이지 저장소에 이슈</a>를 남겨주세요!
</div>

<p class="text-muted text-right mb-0">
  <small><strong>Last Update:</strong> 2026-07-09</small>
</p>
