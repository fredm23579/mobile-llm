! fast_math.f90
! Pure Fortran 2003 implementation of the O(N) linear state-space decay.
! Leveraging Fortran's native array handling to compile down to raw, 
! unabstracted SIMD instructions for maximum mobile CPU performance.

subroutine update_hidden_state(d_model, decay, hidden_state, token_emb) bind(c, name="update_hidden_state")
    use iso_c_binding, only: c_int, c_float
    implicit none
    
    integer(c_int), value :: d_model
    real(c_float), value :: decay
    real(c_float), dimension(d_model), intent(inout) :: hidden_state
    real(c_float), dimension(d_model), intent(in) :: token_emb
    
    integer :: i
    
    ! Force SIMD vectorization on mobile ARM architectures
    !DIR$ SIMD
    do i = 1, d_model
        hidden_state(i) = (decay * hidden_state(i)) + token_emb(i)
    end do

end subroutine update_hidden_state
