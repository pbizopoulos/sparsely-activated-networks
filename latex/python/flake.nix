{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    check-python-script.url = "github:pbizopoulos/check-python-script/main?dir=python";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      check-python-script,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        wfdb = pkgs.python3Packages.buildPythonPackage rec {
          pname = "wfdb";
          version = "4.1.2";
          format = "wheel";
          src = pkgs.python3Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "V+9kMJ7HeTuxFhHGRtEp8LGTa1mJR/7YwU+Qo1ZlBT0=";
            dist = python;
            python = "py3";
          };
          propagatedBuildInputs = [
            pkgs.python3Packages.matplotlib
            pkgs.python3Packages.pandas
            pkgs.python3Packages.scipy
          ];
        };
        packages = with pkgs; [
          python3Packages.torch-bin
          python3Packages.torchvision-bin
          python3Packages.types-requests
          wfdb
        ];
      in
      with pkgs;
      {
        devShells.default = pkgs.mkShell { buildInputs = packages; };
        devShells.check = pkgs.mkShell {
          buildInputs = packages ++ [
            check-python-script.packages.${system}.default
            djlint
            mypy
            nixfmt-rfc-style
            ruff
          ];
        };
      }
    );
}
