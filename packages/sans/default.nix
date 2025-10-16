{
  pkgs ? import <nixpkgs> { },
}:
pkgs.python312Packages.buildPythonPackage rec {
  installPhase = ''
    mkdir -p $out/${pkgs.python312.sitePackages}
    cp ./main.py $out/${pkgs.python312.sitePackages}/${pname}.py
  '';
  pname = builtins.baseNameOf src;
  propagatedBuildInputs = [ pkgs.python312Packages.torch-bin ];
  pyproject = false;
  src = ./.;
  version = "0.0.0";
}
