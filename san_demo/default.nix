{ inputs, pkgs, ... }:
(inputs.afairesi or inputs.self).lib.mkHtmlPackage {
  inherit pkgs;
  meta.description = "An HTML, CSS, and JavaScript template package.";
  src = ./.;
}
