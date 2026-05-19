OPENQASM 2.0;
include "qelib1.inc";
qreg q407[5];
cx q407[1],q407[0];
cx q407[3],q407[4];
cx q407[3],q407[2];
cx q407[2],q407[1];
cx q407[0],q407[1];
