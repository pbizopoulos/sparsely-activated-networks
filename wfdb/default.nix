{ pkgs }:
let
  physionet = python.pkgs.buildPythonPackage rec {
    format = "wheel";
    pname = "physionet";
    propagatedBuildInputs = [
      python.pkgs.boto3
      python.pkgs.fsspec
      python.pkgs.numpy
      python.pkgs.openpyxl
      python.pkgs.pandas
      python.pkgs.requests
      python.pkgs.tqdm
      python.pkgs.urllib3
    ];
    pythonImportsCheck = [ pname ];
    src = pkgs.fetchPypi {
      inherit pname version format;
      dist = "py3";
      hash = "sha256-WT3COUrVxzu7ByYC1LdD9OH20guR1S1E914LJs1X5j4=";
      python = "py3";
    };
    version = "0.7.3";
  };
  python = pkgs.python3;
in
python.pkgs.buildPythonPackage rec {
  doInstallCheck = true;
  format = "wheel";
  installCheckPhase = ''
    runHook preInstallCheck
    PYTHONPATH="$out/${python.sitePackages}:$PYTHONPATH" ${python.interpreter} -m pytest -q -p no:cacheprovider --import-mode=importlib ${./test_main.py}
    runHook postInstallCheck
  '';
  meta.description = "Read and write physiological waveform and annotation data in Python";
  nativeInstallCheckInputs = [ python.pkgs.pytest ];
  pname = "wfdb";
  propagatedBuildInputs = [
    physionet
    python.pkgs.aiohttp
    python.pkgs.fsspec
    python.pkgs.matplotlib
    python.pkgs.numpy
    python.pkgs.pandas
    python.pkgs.requests
    python.pkgs.scipy
    python.pkgs.soundfile
  ];
  pythonImportsCheck = [ pname ];
  src = pkgs.fetchPypi {
    inherit pname version format;
    dist = "py3";
    hash = "sha256-U99omPlvf3FsPjwNuBgMMCNDRgrBkoUCxRftGJe5IFQ=";
    python = "py3";
  };
  version = "5.2.0";
}
