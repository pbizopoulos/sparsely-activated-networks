{ pkgs }:
let
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
  postInstall = ''
    substituteInPlace "$out/${python.sitePackages}/wfdb/io/annotation.py" \
      --replace-fail '.values, inplace=True' '.to_numpy(), inplace=True'
  '';
  propagatedBuildInputs = [
    python.pkgs.aiohttp
    python.pkgs.fsspec
    python.pkgs.matplotlib
    python.pkgs.pandas
    python.pkgs.requests
    python.pkgs.scipy
    python.pkgs.soundfile
  ];
  pythonImportsCheck = [ pname ];
  src = python.pkgs.fetchPypi rec {
    inherit pname version format;
    dist = python;
    python = "py3";
    sha256 = "qhgBz4NXl7kFGreVX7kAzmxfHhts2cyZeebHME4VcGM=";
  };
  version = "4.3.1";
}
