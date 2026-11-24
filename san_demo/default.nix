{ inputs, pkgs, ... }:
inputs.afairesi.lib.mkHtmlPackage {
  inherit pkgs;
  meta.description = "Interactive demonstration of sparsely activated networks and signal reconstruction";
  passthru.tests.page =
    pkgs.runCommand "san_demo-page"
      {
        PAGE = ./index.html;
        nativeBuildInputs = [
          (pkgs.python3.withPackages (ps: [
            ps.beautifulsoup4
            ps.pytest
          ]))
        ];
      }
      ''
        python -m pytest -q -p no:cacheprovider --import-mode=importlib ${./test_main.py}
        touch "$out"
      '';
  src = ./.;
}
