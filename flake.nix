{
  inputs = {
    canonical = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:pbizopoulos/canonical";
    };
    nixpkgs.url = "github:NixOS/nixpkgs/7a0f122f5090cf4c2ade2a13a0e229d4e19ba71f";
  };
  outputs =
    inputs:
    inputs.canonical.blueprint {
      inherit inputs;
      nixpkgs.config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    }
    // {
      inherit (inputs.canonical) formatter;
    };
}
