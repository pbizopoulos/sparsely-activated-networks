{ inputs, pkgs, ... }:
let
  python = pkgs.python3;
in
inputs.afairesi.lib.mkPythonPackage {
  inherit pkgs;
  executable = true;
  makeWrapperArgs = [
    "--prefix"
    "PATH"
    ":"
    (pkgs.lib.makeBinPath [ pkgs.texliveFull ])
  ];
  meta = {
    description = "Sparsely activated networks";
    homepage = "https://arxiv.org/abs/1907.06592";
    longDescription = ''
      Publication: https://ieeexplore.ieee.org/abstract/document/9072295/
      Project website: https://pbizopoulos.github.io/sparsely-activated-networks/
    '';
  };
  nativeCheckInputs = [
    pkgs.texliveFull
    python.pkgs.pandas-stubs
    python.pkgs.scipy-stubs
  ];
  propagatedBuildInputs = [
    inputs.self.packages.${pkgs.stdenv.hostPlatform.system}.wfdb
    python.pkgs.pandas
    python.pkgs.torch
    python.pkgs.torchvision
  ];
  src = ./.;
}
